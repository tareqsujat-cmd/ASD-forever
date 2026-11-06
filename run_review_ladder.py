"""
Sequential ladder for the calibration, interpretation and nesting experiments.

Guarantees
----------
* One stage at a time.  A stage starts only when every stage before it is DONE.
* A stage is DONE only when its command exits with status 0 AND its output validator passes.
  A silent partial result therefore cannot let a later stage start.
* A failing stage is retried (stages are resumable: finished fold or split files are skipped),
  with fewer workers on each retry to relieve memory pressure.  After the last retry the ladder
  HALTS and never starts a later stage.
* A stage whose log has not grown for ``stall_s`` seconds is killed and treated as a failure.
* A job that was already running when the ladder started (the pooled refit) is adopted: the
  ladder waits for it, then validates its output, and re-runs the stage only if needed.
* Everything is logged under results/latest_study/review/ladder/:
      ladder.log        human-readable master log
      events.jsonl      one JSON line per transition (the watcher reads this)
      status.json       current state of every stage, progress, durations, validation messages
      logs/NN_name.log  stdout and stderr of every attempt of every stage
      outputs_manifest.json   size and SHA-256 of every output file, written at the end
      RUN_REPORT.md     stage table with durations and validation results, written at the end

Usage
-----
    .venv/bin/python run_review_ladder.py                  # run (resumes; DONE stages are re-validated)
    .venv/bin/python run_review_ladder.py --dry-run        # list the stages and check preconditions
    .venv/bin/python run_review_ladder.py --from loso_refit
    .venv/bin/python run_review_ladder.py --selftest       # prove that a failure halts the ladder
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / "results" / "latest_study"
REVIEW = STUDY / "review"
PY = sys.executable

Validator = Callable[[], Tuple[bool, str]]


# =========================================================================================
# paths that depend on where the ladder keeps its state (overridden by --selftest)
# =========================================================================================
class Paths:
    def __init__(self, base: Path):
        self.base = base
        self.logs = base / "logs"
        self.status = base / "status.json"
        self.events = base / "events.jsonl"
        self.master = base / "ladder.log"
        self.lock = base / "ladder.pid"
        self.manifest = base / "outputs_manifest.json"
        self.report = base / "RUN_REPORT.md"


P = Paths(REVIEW / "ladder")


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    line = f"{now()}  {msg}"
    print(line, flush=True)
    with open(P.master, "a") as f:
        f.write(line + "\n")


def event(stage: str, kind: str, **detail) -> None:
    rec = dict(ts=now(), stage=stage, event=kind, **detail)
    with open(P.events, "a") as f:
        f.write(json.dumps(rec) + "\n")


# =========================================================================================
# stage definition
# =========================================================================================
@dataclass
class Stage:
    name: str
    desc: str
    cmd: Optional[Callable[[int], List[str]]] = None       # attempt index -> command (None: check only)
    validate: Optional[Validator] = None
    retries: int = 2
    stall_s: int = 3600
    exit_grace_s: int = 0                                  # >0: a hung process with complete, valid output is ended after this idle time
    adopt: Optional[str] = None                            # pgrep pattern of an already running job
    adopt_log: Optional[Path] = None
    progress: Optional[Callable[[], str]] = None
    heavy: bool = True
    state: str = "PENDING"
    info: dict = field(default_factory=dict)


# =========================================================================================
# validators
# =========================================================================================
def _count(pattern_dir: Path, pattern: str) -> int:
    return len([f for f in pattern_dir.glob(pattern) if ".tmp" not in f.name])


def v_folds(protocol: str, n: int) -> Validator:
    def check():
        import numpy as np
        d = REVIEW / "nested"
        files = [f for f in sorted(d.glob(f"fold_{protocol}_*.npz")) if ".tmp" not in f.name]
        if len(files) != n:
            return False, f"{len(files)} of {n} fold files"
        need = {"te_idx", "tr_idx", "p_raw", "p_unpen", "inner_p", "coef", "struct", "z_te",
                "c_best", "intercept", "score_sd"}
        for f in files:
            z = np.load(f)
            if not need <= set(z.files):
                return False, f"{f.name} lacks {sorted(need - set(z.files))}"
            if not np.isfinite(z["inner_p"]).all():
                return False, f"{f.name}: non-finite inner predictions"
            p = z["p_raw"]
            if not ((p > 0) & (p < 1)).all():
                return False, f"{f.name}: probabilities outside (0, 1)"
        sys.path.insert(0, str(ROOT))
        import exp_calibration_nested as cal
        try:
            _, _, _, _, chk = cal.assemble(protocol)
        except BaseException as e:                         # SystemExit from a failed alignment check
            return False, f"reproduction check failed: {e}"
        if chk["max_abs_diff_native"] > 1e-5:
            return False, f"refit differs from stored predictions by {chk['max_abs_diff_native']:.2e}"
        return True, (f"{n} folds; refit reproduces the stored predictions to "
                      f"{chk['max_abs_diff_native']:.1e} on {chk['n_heldout']} participants")
    return check


def v_null() -> Tuple[bool, str]:
    import numpy as np
    files = [f for f in sorted((REVIEW / "nested_null").glob("null_pooled_*.npz")) if ".tmp" not in f.name]
    if len(files) != 10:
        return False, f"{len(files)} of 10 null files"
    for f in files:
        z = np.load(f)
        if z["coef"].shape[0] != z["struct"].shape[0] or z["coef"].shape[1] != 3:
            return False, f"{f.name}: unexpected shapes"
        if not (np.isfinite(z["coef"]).all() and np.isfinite(z["struct"]).all()
                and np.isfinite(z["partial"]).all()):
            return False, f"{f.name}: non-finite values"
    n_perm = np.load(files[0])["coef"].shape[0]
    return True, f"10 null folds with {n_perm} permutations each"


def v_splithalf() -> Tuple[bool, str]:
    import numpy as np
    files = [f for f in sorted((REVIEW / "splithalf").glob("split_*.npz")) if ".tmp" not in f.name]
    if len(files) != 40:
        return False, f"{len(files)} of 40 split files"
    for f in files:
        z = np.load(f)
        if not np.isfinite(z["stats"]).all():
            return False, f"{f.name}: non-finite statistics"
    return True, f"40 splits, {np.load(files[0])['stats'].shape[-1] - 1} permutations each"


def v_csvs(names: List[str], rows: Optional[dict] = None, directory: Path = REVIEW) -> Validator:
    def check():
        import pandas as pd
        for n in names:
            p = directory / n
            if not p.exists() or p.stat().st_size == 0:
                return False, f"{n} missing or empty"
            df = pd.read_csv(p)
            if len(df) == 0:
                return False, f"{n} has no rows"
            if rows and n in rows and len(df) != rows[n]:
                return False, f"{n}: {len(df)} rows, expected {rows[n]}"
        return True, "; ".join(f"{n} ({len(pd.read_csv(directory / n))} rows)" for n in names)
    return check


def v_calibration(directory: Path, protocols: int) -> Validator:
    def check():
        ok, msg = v_csvs(["calibration_nested.csv", "calibration_nested_paired.csv",
                          "calibration_reliability.csv", "calibration_per_site.csv",
                          "calibration_site_heterogeneity.csv",
                          "calibration_recalibration_params.csv"],
                         {"calibration_nested.csv": protocols * 5 * 12,
                          "calibration_nested_paired.csv": protocols * 4 * 12}, directory)()
        if not ok:
            return ok, msg
        s = json.loads((directory / "calibration_nested_summary.json").read_text())
        worst = max(v["max_abs_diff_native"] for v in s["reproduction_check"].values())
        if worst > 1e-5:
            return False, f"native probabilities differ from stored ones by {worst:.2e}"
        return True, f"{protocols} protocol(s); reproduction error {worst:.1e}"
    return check


def v_interpretation(directory: Path, protocols: List[str]) -> Validator:
    def check():
        ok, msg = v_csvs(["interpretation_blocks.csv", "interpretation_stability.csv",
                          "interpretation_crossview.csv", "interpretation_concordance.csv",
                          "interpretation_deletion.csv"], None, directory)()
        if not ok:
            return ok, msg
        c = json.loads((directory / "interpretation_checks.json").read_text())["checks"]
        for p in protocols:
            if p not in c:
                return False, f"checks for {p} missing"
            if c[p]["max_additivity_error"] > 1e-9:
                return False, f"{p}: contributions do not add ({c[p]['max_additivity_error']:.1e})"
            if c[p]["max_prob_reproduction_error"] > 1e-6:
                return False, f"{p}: weights do not reproduce probabilities ({c[p]['max_prob_reproduction_error']:.1e})"
        return True, "; ".join(f"{p}: additivity {c[p]['max_additivity_error']:.0e}, "
                               f"reproduction {c[p]['max_prob_reproduction_error']:.0e}" for p in protocols)
    return check


def v_fully_nested(protocol: str, n: int) -> Validator:
    def check():
        files = [f for f in sorted((REVIEW / "fully_nested").glob(f"fn_{protocol}_*.npz")) if ".tmp" not in f.name]
        if len(files) != n:
            return False, f"{len(files)} of {n} files"
        import numpy as np
        changed = sum(int(np.load(f)["changed"].sum()) for f in files)
        ext = sum(int(np.load(f)["changed_ext"].sum()) for f in files)
        return True, (f"{n} folds; penalty changed in {changed} of {3 * n} cases on the published grid, "
                      f"{ext} on the extended grid")
    return check


def v_files(paths: List[Path], min_bytes: int = 2000) -> Validator:
    def check():
        for p in paths:
            if not p.exists() or p.stat().st_size < min_bytes:
                return False, f"{p.name} missing or too small"
        return True, f"{len(paths)} files"
    return check


def v_always() -> Tuple[bool, str]:
    return True, "no output to validate"


# =========================================================================================
# progress shown in status.json and by the watcher
# =========================================================================================
def prog(pattern_dir: Path, pattern: str, total: int) -> Callable[[], str]:
    return lambda: f"{_count(pattern_dir, pattern)}/{total}"


# =========================================================================================
# the ladder
# =========================================================================================
def build_stages() -> List[Stage]:
    nested, rv = REVIEW / "nested", REVIEW
    preview = REVIEW / "preview_pooled"

    def workers(base):                                     # fewer workers on each retry
        return lambda a: max(2, base - a)

    def nested_cmd(protocol, base=4):
        w = workers(base)
        return lambda a: [PY, "exp_nested_outer.py", "--protocol", protocol, "--workers", str(w(a))]

    S: List[Stage] = []
    S.append(Stage("preflight", "environment, inputs, disk, script syntax; moves stale partial outputs aside",
                   cmd=None, validate=None, heavy=False))
    S.append(Stage("pooled_refit", "pooled ten-fold refit with inner cross-fitting (adopts a running job)",
                   cmd=nested_cmd("pooled"), validate=v_folds("pooled", 10), retries=2, stall_s=2700, exit_grace_s=240,
                   adopt=r"^\S*python\S*\s+exp_nested_outer\.py\s+--protocol\s+pooled\b", adopt_log=nested / "run_pooled.log",
                   progress=prog(nested, "fold_pooled_*.npz", 10)))
    S.append(Stage("pooled_preview_calibration", "pooled-only calibration analysis (preview, separate directory)",
                   cmd=lambda a: [PY, "exp_calibration_nested.py", "--protocols", "pooled",
                                  "--reps", "2000", "--out", str(preview)],
                   validate=v_calibration(preview, 1), retries=1, stall_s=5400))
    S.append(Stage("pooled_preview_interpretation", "pooled-only interpretation analysis (preview)",
                   cmd=lambda a: [PY, "exp_interpretation.py", "--protocols", "pooled",
                                  "--reps", "2000", "--out", str(preview)],
                   validate=v_interpretation(preview, ["pooled"]), retries=1, stall_s=5400))
    S.append(Stage("null_permutations", "100 within-site label permutations, pooled ten-fold",
                   cmd=lambda a: [PY, "exp_interpretation_null.py", "--workers", str(workers(4)(a)),
                                  "--n-perm", "100"],
                   validate=v_null, retries=2, stall_s=2700, exit_grace_s=240,
                   progress=prog(REVIEW / "nested_null", "null_pooled_*.npz", 10)))
    S.append(Stage("splithalf", "40 independent split-halves with 30 permutations each",
                   cmd=lambda a: [PY, "exp_interpretation_splithalf.py", "--workers", str(workers(4)(a)),
                                  "--splits", "40", "--n-perm", "30"],
                   validate=v_splithalf, retries=2, stall_s=2700, exit_grace_s=240,
                   progress=prog(REVIEW / "splithalf", "split_*.npz", 40)))
    S.append(Stage("fully_nested_pooled", "fully nested selection of the penalty, pooled",
                   cmd=lambda a: [PY, "exp_fully_nested_c.py", "--protocol", "pooled",
                                  "--workers", str(workers(4)(a))],
                   validate=v_fully_nested("pooled", 10), retries=2, stall_s=2700, exit_grace_s=240,
                   progress=prog(REVIEW / "fully_nested", "fn_pooled_*.npz", 10)))
    S.append(Stage("loso_refit", "leave-one-site-out refit with inner cross-fitting",
                   cmd=nested_cmd("loso"), validate=v_folds("loso", 20), retries=2, stall_s=2700, exit_grace_s=240,
                   progress=prog(nested, "fold_loso_*.npz", 20)))
    S.append(Stage("fully_nested_loso", "fully nested selection of the penalty, leave-one-site-out",
                   cmd=lambda a: [PY, "exp_fully_nested_c.py", "--protocol", "loso",
                                  "--workers", str(workers(4)(a))],
                   validate=v_fully_nested("loso", 20), retries=2, stall_s=2700, exit_grace_s=240,
                   progress=prog(REVIEW / "fully_nested", "fn_loso_*.npz", 20)))
    S.append(Stage("final_calibration", "calibration analysis, both protocols",
                   cmd=lambda a: [PY, "exp_calibration_nested.py", "--reps", "2000"],
                   validate=v_calibration(rv, 2), retries=1, stall_s=5400))
    S.append(Stage("final_interpretation", "interpretation analysis, both protocols",
                   cmd=lambda a: [PY, "exp_interpretation.py", "--reps", "2000"],
                   validate=v_interpretation(rv, ["pooled", "loso"]), retries=1, stall_s=5400))
    S.append(Stage("null_analysis", "permutation reference compared with the real fits",
                   cmd=lambda a: [PY, "exp_interpretation_null.py", "--analyse"],
                   validate=v_csvs(["interpretation_stability_null.csv", "interpretation_block_null.csv"],
                                   {"interpretation_stability_null.csv": 30,
                                    "interpretation_block_null.csv": 4 * 36}),
                   retries=1, stall_s=3600, heavy=False))
    S.append(Stage("splithalf_analysis", "split-half replication compared with its permutation null",
                   cmd=lambda a: [PY, "exp_interpretation_splithalf.py", "--analyse"],
                   validate=v_csvs(["interpretation_splithalf.csv"], {"interpretation_splithalf.csv": 30}),
                   retries=1, stall_s=3600, heavy=False))
    S.append(Stage("fully_nested_summary", "summary of the fully nested penalty selection",
                   cmd=lambda a: [PY, "exp_fully_nested_c.py", "--summarise"],
                   validate=v_csvs(["fully_nested_c_selection.csv", "fully_nested_c_summary.csv"],
                                   {"fully_nested_c_summary.csv": 2}),
                   retries=1, stall_s=1800, heavy=False))
    S.append(Stage("figures", "calibration and interpretation figures",
                   cmd=lambda a: [PY, "viz_calibration_interpretation.py"],
                   validate=v_files([STUDY / "figures" / "fig_calibration.png",
                                     STUDY / "figures" / "fig_interpretation.png"], 20000),
                   retries=1, stall_s=1800, heavy=False))
    S.append(Stage("report", "output manifest and run report", cmd=None, validate=None, heavy=False))
    return S


# =========================================================================================
# preflight
# =========================================================================================
def preflight() -> Tuple[bool, str]:
    msgs = []
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < 10:
        return False, f"only {free_gb:.1f} GB free"
    msgs.append(f"{free_gb:.0f} GB free")
    for f in ("abide_processed/mri/metadata.csv", "atlas/roi_table.csv",
              "results/latest_study/raw/voting.json", "results/latest_study/cache/nested/labels.npz",
              "results/latest_study/tables/network_contrast.csv",
              "results/latest_study/tables/sensorimotor_regions.csv"):
        if not (ROOT / f).exists():
            return False, f"missing input {f}"
    scripts = ["exp_nested_outer.py", "exp_calibration_nested.py", "exp_interpretation.py",
               "exp_interpretation_null.py", "exp_interpretation_splithalf.py",
               "exp_fully_nested_c.py", "viz_calibration_interpretation.py"]
    for s in scripts:
        r = subprocess.run([PY, "-m", "py_compile", str(ROOT / s)], capture_output=True, text=True)
        if r.returncode:
            return False, f"{s} does not compile: {r.stderr.strip()[:200]}"
    msgs.append(f"{len(scripts)} scripts compile")
    import numpy, pandas, scipy, sklearn
    msgs.append(f"python {platform.python_version()}, numpy {numpy.__version__}, "
                f"scikit-learn {sklearn.__version__}")
    # partial outputs of earlier smoke tests are moved aside so that nothing stale is mistaken for a result
    aside = P.base / "_superseded_smoke"
    moved = []
    prev = json.loads(P.status.read_text()) if P.status.exists() else {"stages": []}
    final_done = any(x["name"] in ("final_calibration", "final_interpretation") and x["state"] == "DONE"
                     for x in prev["stages"])
    for pat in () if final_done else ("calibration_*.csv", "calibration_*.json", "interpretation_*.csv",
                "interpretation_*.json", "interpretation_*.npz"):
        for f in REVIEW.glob(pat):
            aside.mkdir(exist_ok=True)
            shutil.move(str(f), str(aside / f"{time.strftime('%H%M%S')}_{f.name}"))
            moved.append(f.name)
    if moved:
        msgs.append(f"moved {len(moved)} stale partial outputs to {aside.name}/")
    return True, "; ".join(msgs)


# =========================================================================================
# running a stage
# =========================================================================================
def save_status(stages: List[Stage], overall: str, current: Optional[str]) -> None:
    doc = dict(updated=now(), overall=overall, current=current, pid=os.getpid(),
               stages=[dict(name=s.name, desc=s.desc, state=s.state, **s.info) for s in stages])
    tmp = P.status.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, P.status)


def pids_matching(pattern: str) -> List[int]:
    """Processes whose command line matches the regular expression ``pattern``.

    The pattern is matched against the whole command line, so it should be anchored at the start
    with the interpreter (for example r"^\\S*python\\S*\\s+exp_nested_outer\\.py"), which excludes
    shells and editors that merely mention the script name.  The ladder's own ancestors are never
    returned.
    """
    import re
    me = {os.getpid()}
    pid = os.getppid()
    while pid > 1 and pid not in me:                       # every ancestor of the ladder
        me.add(pid)
        r = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True)
        try:
            pid = int(r.stdout.strip())
        except ValueError:
            break
    rx = re.compile(pattern, re.IGNORECASE)
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(" ")
        if pid_s.isdigit() and int(pid_s) not in me and rx.search(cmd.strip()):
            found.append(int(pid_s))
    return found


def wait_adopted(stage: Stage, stages: List[Stage]) -> None:
    pids = pids_matching(stage.adopt)
    if not pids:
        return
    log(f"[{stage.name}] adopting a running job (pids {pids}); waiting for it to finish")
    event(stage.name, "adopt", pids=pids)
    last_change, last_size = time.time(), -1
    while pids_matching(stage.adopt):
        time.sleep(20)
        if stage.adopt_log and stage.adopt_log.exists():
            size = stage.adopt_log.stat().st_size
            if size != last_size:
                last_size, last_change = size, time.time()
        if stage.progress:
            stage.info["progress"] = stage.progress()
            save_status(stages, "RUNNING", stage.name)
        if time.time() - last_change > stage.stall_s:
            log(f"[{stage.name}] adopted job stalled for {stage.stall_s}s; killing it")
            for p in pids_matching(stage.adopt):
                os.kill(p, signal.SIGTERM)
            time.sleep(10)
            break
    log(f"[{stage.name}] adopted job has exited")


def run_attempt(stage: Stage, attempt: int, stages: List[Stage], logfile: Path) -> int:
    cmd = stage.cmd(attempt)
    with open(logfile, "a") as lf:
        lf.write(f"\n===== attempt {attempt + 1}  {now()} =====\n$ {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)
        log(f"[{stage.name}] attempt {attempt + 1}: pid {proc.pid}: {' '.join(cmd[1:])}")
        last_size, last_change, last_beat, last_val = -1, time.time(), 0.0, 0.0
        while proc.poll() is None:
            time.sleep(15)
            size = logfile.stat().st_size
            if size != last_size:
                last_size, last_change = size, time.time()
            if stage.progress:
                stage.info["progress"] = stage.progress()
            if time.time() - last_beat > 120:
                stage.info["heartbeat"] = now()
                save_status(stages, "RUNNING", stage.name)
                last_beat = time.time()
            if (stage.exit_grace_s and stage.validate and time.time() - last_change > stage.exit_grace_s
                    and time.time() - last_val > 120):
                # the output may already be complete while the process hangs at shutdown (idle children that
                # are never joined); a complete, validated output is what matters, so the process is ended
                last_val = time.time()
                try:
                    v_ok, v_msg = stage.validate()
                except BaseException:
                    v_ok, v_msg = False, ""
                if v_ok:
                    idle = int(time.time() - last_change)
                    log(f"[{stage.name}] output is complete and valid but the process has been idle for {idle}s "
                        f"and has not exited; terminating it ({v_msg})")
                    event(stage.name, "exit_hang_kill", idle_seconds=idle, detail=v_msg)
                    os.killpg(proc.pid, signal.SIGTERM)
                    time.sleep(10)
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                    lf.write("\n[ladder] terminated after completing: the process hung at exit\n")
                    return 0
            if time.time() - last_change > stage.stall_s:
                log(f"[{stage.name}] no log output for {stage.stall_s}s; terminating the stage")
                event(stage.name, "stall_kill", seconds=stage.stall_s)
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(15)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                lf.write("\n[ladder] terminated after a stall\n")
                return -9
        return proc.returncode


def tail(path: Path, n: int = 12) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(True)[-n:])
    except OSError:
        return ""


def run_stage(stage: Stage, stages: List[Stage], idx: int, force: bool = False) -> bool:
    if stage.validate and not force and stage.cmd is not None:
        try:
            ok, msg = stage.validate()
        except BaseException as e:                         # a validator must never crash the ladder
            ok, msg = False, f"validator error: {e}"
        if ok:
            stage.state = "DONE"
            stage.info.update(ended=now(), seconds=0, attempts=0, validation=f"already complete: {msg}")
            save_status(stages, "RUNNING", stage.name)
            event(stage.name, "skip_valid", detail=msg)
            log(f"[{stage.name}] already complete and valid, not re-run: {msg}")
            return True
    stage.state = "RUNNING"
    lp = P.logs / f"{idx:02d}_{stage.name}.log"
    try:
        lp_show = str(lp.relative_to(ROOT))
    except ValueError:                                     # the self-test keeps its state outside the project
        lp_show = str(lp)
    stage.info.update(started=now(), attempts=0, log=lp_show)
    t0 = time.time()
    save_status(stages, "RUNNING", stage.name)
    event(stage.name, "start", desc=stage.desc)
    log(f"[{stage.name}] START  {stage.desc}")
    logfile = P.logs / f"{idx:02d}_{stage.name}.log"

    if stage.name == "preflight":
        ok, msg = preflight()
    elif stage.name == "report":
        ok, msg = write_report(stages)
    else:
        ok, msg = False, "not run"
        if stage.adopt:
            wait_adopted(stage, stages)
            if stage.validate:
                try:
                    ok, msg = stage.validate()
                except BaseException as e:
                    ok, msg = False, f"validator error: {e}"
                if ok:
                    log(f"[{stage.name}] the adopted job left a valid result: {msg}")
        attempt = 0
        while not ok and attempt <= stage.retries:
            stage.info["attempts"] = attempt + 1
            rc = run_attempt(stage, attempt, stages, logfile)
            try:
                v_ok, v_msg = stage.validate() if stage.validate else (True, "no validator")
            except BaseException as e:
                v_ok, v_msg = False, f"validator error: {e}"
            ok = (rc == 0) and v_ok
            msg = f"exit {rc}; validation: {v_msg}"
            event(stage.name, "attempt_end", attempt=attempt + 1, exit=rc, valid=v_ok, detail=v_msg)
            log(f"[{stage.name}] attempt {attempt + 1} finished: {msg}")
            if not ok:
                log(f"[{stage.name}] last log lines:\n{tail(logfile)}")
                attempt += 1
                if attempt <= stage.retries:
                    log(f"[{stage.name}] retrying in 60 s (finished files are kept)")
                    time.sleep(60)
    stage.info.update(ended=now(), seconds=round(time.time() - t0), validation=msg)
    if stage.progress:
        stage.info["progress"] = stage.progress()
    stage.state = "DONE" if ok else "FAILED"
    save_status(stages, "RUNNING", stage.name)
    event(stage.name, "done" if ok else "failed", seconds=round(time.time() - t0), detail=msg)
    log(f"[{stage.name}] {'DONE' if ok else 'FAILED'} in {time.time() - t0:.0f}s: {msg}")
    return ok


# =========================================================================================
# final report and manifest
# =========================================================================================
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_report(stages: List[Stage]) -> Tuple[bool, str]:
    outs = {}
    for pat in ("*.csv", "*.json", "*.npz"):
        for f in sorted(REVIEW.glob(pat)):
            outs[str(f.relative_to(ROOT))] = dict(bytes=f.stat().st_size, sha256=sha256(f))
    for sub in ("preview_pooled", "fully_nested", "splithalf", "nested_null", "nested"):
        d = REVIEW / sub
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.is_file() and ".tmp" not in f.name:
                    outs[str(f.relative_to(ROOT))] = dict(bytes=f.stat().st_size, sha256=sha256(f))
    for f in sorted((STUDY / "figures").glob("fig_calibration.*")) + sorted((STUDY / "figures").glob("fig_interpretation.*")):
        outs[str(f.relative_to(ROOT))] = dict(bytes=f.stat().st_size, sha256=sha256(f))
    import numpy, pandas, scipy, sklearn, torch
    env = dict(python=platform.python_version(), platform=platform.platform(),
               numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__,
               sklearn=sklearn.__version__, torch=torch.__version__)
    P.manifest.write_text(json.dumps(dict(written=now(), environment=env, n_files=len(outs),
                                          files=outs), indent=2))
    lines = ["# Review ladder: run report", "", f"Written {now()}.", "",
             "| # | stage | state | minutes | attempts | validation |", "|---|---|---|---|---|---|"]
    for i, s in enumerate(stages):
        if s.name == "report":
            continue
        lines.append(f"| {i} | {s.name} | {s.state} | {s.info.get('seconds', 0) / 60:.1f} | "
                     f"{s.info.get('attempts', 0)} | {str(s.info.get('validation', '')).replace('|', '/')} |")
    lines += ["", f"{len(outs)} output files are listed with size and SHA-256 in `outputs_manifest.json`.", "",
              "Environment: " + ", ".join(f"{k} {v}" for k, v in env.items())]
    P.report.write_text("\n".join(lines) + "\n")
    return True, f"{len(outs)} files hashed; report written"


# =========================================================================================
# driver
# =========================================================================================
def acquire_lock() -> None:
    if P.lock.exists():
        try:
            old = int(P.lock.read_text())
            os.kill(old, 0)
            raise SystemExit(f"another ladder is running (pid {old})")
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    P.lock.write_text(str(os.getpid()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from", dest="start", default=None, help="first stage to run")
    ap.add_argument("--only", default=None, help="run a single stage")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    global P
    if args.selftest:
        return selftest()

    P.logs.mkdir(parents=True, exist_ok=True)
    stages = build_stages()
    if args.dry_run:
        for i, s in enumerate(stages):
            state = ""
            if s.validate:
                try:
                    v_ok, v_msg = s.validate()
                except BaseException as e:
                    v_ok, v_msg = False, f"validator error: {e}"
                state = f"[{'valid' if v_ok else 'not yet'}: {v_msg[:70]}]"
            print(f"{i:2d}  {s.name:30s} {s.desc[:58]:58s} {state}")
        running = pids_matching(r"^\S*python\S*\s+exp_nested_outer\.py")
        print(f"running refit processes: {running or 'none'}")
        return
    acquire_lock()
    try:
        names = [s.name for s in stages]
        first = names.index(args.start) if args.start else 0
        log("=" * 88)
        log(f"ladder started, pid {os.getpid()}, {len(stages)} stages, from '{names[first]}'")
        event("ladder", "start", stages=names, first=names[first])
        prev = json.loads(P.status.read_text()) if P.status.exists() else None
        done_before = {s["name"] for s in prev["stages"] if s["state"] == "DONE"} if prev else set()
        for i, s in enumerate(stages):
            if args.only and s.name != args.only:
                s.state = "SKIPPED"
                continue
            if i < first:
                s.state = "SKIPPED"
                continue
            ok = run_stage(s, stages, i, force=bool(args.only or (args.start and i == first)))
            if not ok:
                for later in stages[i + 1:]:
                    later.state = "NOT STARTED"
                save_status(stages, "HALTED", s.name)
                event("ladder", "halt", failed=s.name)
                log(f"HALTED at '{s.name}'; no later stage was started. Fix the cause and re-run "
                    f"the ladder; finished files are kept and DONE stages are re-validated.")
                sys.exit(1)
        save_status(stages, "COMPLETE", None)
        event("ladder", "complete")
        log("ladder COMPLETE: every stage is DONE and validated")
    finally:
        if P.lock.exists() and P.lock.read_text() == str(os.getpid()):
            P.lock.unlink()


# =========================================================================================
# self-test: a failing stage must halt the ladder and later stages must not start
# =========================================================================================
def selftest() -> None:
    global P
    base = Path(os.environ.get("TMPDIR", "/tmp")) / f"ladder_selftest_{os.getpid()}"
    P = Paths(base)
    P.logs.mkdir(parents=True, exist_ok=True)
    marker = base / "C_started"

    def cmd(code: str) -> Callable[[int], List[str]]:
        return lambda a: [PY, "-c", code]

    stages = [
        Stage("A_ok", "succeeds", cmd("print('a')"), lambda: (True, "ok"), retries=0, heavy=False),
        Stage("B_exit_ok_but_invalid", "exits 0 but its validator fails",
              cmd("print('b')"), lambda: (False, "output incomplete"), retries=1, heavy=False),
        Stage("C_must_not_start", "must never run",
              cmd(f"open({str(marker)!r}, 'w').write('x')"), lambda: (True, "ok"), retries=0, heavy=False),
    ]
    import builtins
    orig_sleep = time.sleep
    time.sleep = lambda s: orig_sleep(min(s, 0.2))          # do not wait 60 s for the retry in a test
    try:
        for i, s in enumerate(stages):
            if not run_stage(s, stages, i):
                for later in stages[i + 1:]:
                    later.state = "NOT STARTED"
                save_status(stages, "HALTED", s.name)
                break
    finally:
        time.sleep = orig_sleep
    states = {s.name: s.state for s in stages}
    print("states:", states)
    assert states["A_ok"] == "DONE", states
    assert states["B_exit_ok_but_invalid"] == "FAILED", states
    assert states["C_must_not_start"] == "NOT STARTED" and not marker.exists(), states
    assert json.loads(P.status.read_text())["overall"] == "HALTED"
    print("SELFTEST PASSED: a stage that exits 0 with an invalid output fails, is retried once, and halts "
          "the ladder; the next stage never starts")
    shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()

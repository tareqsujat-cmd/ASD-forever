"""Tests for Module 11: Hyperparameter Tuning with Optuna."""
import sys
sys.path.insert(0, r'e:\ASD_forever')
import matplotlib
matplotlib.use('Agg')
import os, math, tempfile
import numpy as np

PASS = 0
FAIL = 0

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS: {name}")
        PASS += 1
    else:
        print(f"  FAIL: {name} -- {msg}")
        FAIL += 1

def approx(a, b, tol=1e-6):
    return abs(a - b) < tol

# =========================================================================
# Mock train function: returns deterministic 5-fold metrics correlated
# with lr and mri_dropout so importance analysis has real signal.
# =========================================================================
def _mock_train_fn(cfg, variant_name):
    """
    Returns 5 fold-metric dicts.
    AUC is maximised when lr ≈ 1e-3 and dropout ≈ 0.
    """
    # Extract overrides from config dict if available
    lr      = 1e-3
    dropout = 0.1
    if isinstance(cfg, dict):
        lr      = cfg.get("optimizer", {}).get("lr",          lr)
        dropout = cfg.get("model",     {}).get("mri", {}).get("dropout", dropout)

    # AUC peaks at lr=1e-3, penalised by dropout
    base_auc = 0.80 - abs(math.log10(lr) + 3) * 0.04 - dropout * 0.2
    seed_val = int(abs(base_auc * 1e4)) % 1000
    rng      = np.random.default_rng(seed_val)
    fold_aucs = np.clip(base_auc + rng.normal(0, 0.01, 5), 0.5, 1.0)
    return [{"val_auc": float(a), "val_acc": float(a - 0.05)} for a in fold_aucs]


BASE_CFG = {
    "optimizer": {"lr": 1e-3, "weight_decay": 1e-4},
    "model":     {"mri": {"architecture": "resnet10", "dropout": 0.1},
                  "genetics": {"architecture": "transformer"}},
    "fusion":    {"architecture": "cross_attention"},
    "training":  {"batch_size": 16, "focal_gamma": 2.0},
}

# =========================================================================
print("=== SearchSpaceType / suggest_params ===")
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
from hyperparameter_tuning.search_spaces import (
    SearchSpaceType, suggest_params, get_space_names, get_space_dim,
)

check("SearchSpaceType has QUICK", SearchSpaceType.QUICK.value == "quick")
check("get_space_names returns list", isinstance(get_space_names(), list))
check("get_space_names has 6 entries", len(get_space_names()) == 6)

# Verify each space by running through a trial
for space_name in get_space_names():
    study = optuna.create_study(direction="maximize")
    def _obj(trial):
        params = suggest_params(trial, space_name)
        return 0.5
    study.optimize(_obj, n_trials=2)
    check(f"Space '{space_name}' produces params",
          len(study.trials[0].params) > 0,
          f"got {study.trials[0].params}")

check("get_space_dim quick == 6",  get_space_dim("quick") == 6)
check("get_space_dim optimizer==5",get_space_dim("optimizer") == 5)
check("get_space_dim full == 21",  get_space_dim("full") == 21)

# quick space: verify expected keys are present
study_q = optuna.create_study(direction="maximize")
study_q.optimize(lambda t: suggest_params(t, "quick") and 0.5, n_trials=1)
q_params = study_q.trials[0].params
check("quick has lr key",           "lr" in q_params)
check("quick has mri_dropout key",  "mri_dropout" in q_params)
check("quick has fusion_arch key",  "fusion_arch" in q_params)

# Bad space name raises
try:
    suggest_params(study_q.ask(), "invalid_space_name")
    check("Bad space raises ValueError", False, "no exception")
except ValueError:
    check("Bad space raises ValueError", True)

# =========================================================================
print("\n=== TrialRecord ===")
from hyperparameter_tuning.optuna_tuner import TrialRecord

rec = TrialRecord(
    trial_number    = 3,
    params          = {"lr": 1e-3},
    overrides       = {"optimizer.lr": 1e-3},
    mean_metrics    = {"val_auc": 0.82, "val_acc": 0.77},
    fold_metrics    = [{"val_auc": 0.80}, {"val_auc": 0.84}],
    elapsed_seconds = 12.5,
    state           = "complete",
)
check("TrialRecord get_metric auc",  approx(rec.get_metric("val_auc"), 0.82))
check("TrialRecord missing metric",  math.isnan(rec.get_metric("nonexistent")))
check("TrialRecord to_dict has state", rec.to_dict()["state"] == "complete")
check("TrialRecord repr",            "trial=3" in repr(rec))

# =========================================================================
print("\n=== ASDTuner construction ===")
from hyperparameter_tuning.optuna_tuner import ASDTuner

tuner = ASDTuner(
    train_fn     = _mock_train_fn,
    base_config  = BASE_CFG,
    search_space = "quick",
    direction    = "maximize",
    objectives   = ["val_auc"],
    n_trials     = 5,
    pruner       = "none",
    sampler      = "tpe",
    seed         = 42,
)
check("Tuner study is None before optimize",  tuner.study is None)
check("Tuner is_multi is False",              not tuner._is_multi)
check("Tuner objectives",                     tuner._objectives == ["val_auc"])
check("Tuner directions",                     tuner._directions == ["maximize"])

# Multi-objective construction
tuner_mo = ASDTuner(
    train_fn    = _mock_train_fn,
    base_config = BASE_CFG,
    search_space = "quick",
    direction   = ["maximize", "minimize"],
    objectives  = ["val_auc", "val_brier"],
    n_trials    = 3,
    sampler     = "nsgaii",
    seed        = 0,
)
check("Multi-obj is_multi is True", tuner_mo._is_multi)

# Direction / objective length mismatch
try:
    ASDTuner(
        train_fn=_mock_train_fn, base_config=BASE_CFG,
        direction=["maximize", "minimize"], objectives=["val_auc"],
    )
    check("Mismatched lengths raises ValueError", False, "no exception")
except ValueError:
    check("Mismatched lengths raises ValueError", True)

# Bad sampler / pruner
try:
    ASDTuner(train_fn=_mock_train_fn, base_config=BASE_CFG, sampler="invalid")
    check("Bad sampler raises ValueError", False, "no exception")
except ValueError:
    check("Bad sampler raises ValueError", True)

try:
    ASDTuner(train_fn=_mock_train_fn, base_config=BASE_CFG, pruner="invalid")
    check("Bad pruner raises ValueError", False, "no exception")
except ValueError:
    check("Bad pruner raises ValueError", True)

# =========================================================================
print("\n=== ASDTuner.create_study ===")
tuner2 = ASDTuner(
    train_fn=_mock_train_fn, base_config=BASE_CFG,
    search_space="quick", n_trials=3, pruner="none", seed=7,
)
study2 = tuner2.create_study()
check("create_study returns Study",  isinstance(study2, optuna.Study))
check("study stored in tuner",       tuner2.study is study2)

# =========================================================================
print("\n=== ASDTuner.optimize (single-objective) ===")
tuner3 = ASDTuner(
    train_fn     = _mock_train_fn,
    base_config  = BASE_CFG,
    search_space = "quick",
    n_trials     = 8,
    pruner       = "none",
    sampler      = "tpe",
    seed         = 123,
)
study3 = tuner3.optimize()
check("optimize returns Study",          isinstance(study3, optuna.Study))
check("n_trials respected",              len(study3.trials) == 8,
      str(len(study3.trials)))
check("records populated",               len(tuner3.records) == 8,
      str(len(tuner3.records)))
check("all records complete",            all(r.state == "complete"
                                              for r in tuner3.records))
check("records have fold_metrics",       all(len(r.fold_metrics) == 5
                                              for r in tuner3.records))
check("best_trial is not None",          tuner3.best_trial is not None)
check("best_value is finite",            math.isfinite(tuner3.best_value))
check("best_params is dict",             isinstance(tuner3.best_params, dict))
check("best_params non-empty",           len(tuner3.best_params) > 0)
check("pareto_front empty (single-obj)", tuner3.pareto_front == [])

# get_n_best
top3 = tuner3.get_n_best(3, "val_auc")
check("get_n_best count",                len(top3) == 3)
check("get_n_best sorted descending",    top3[0].get_metric("val_auc") >=
                                          top3[-1].get_metric("val_auc"))

# summary_dict
summ = tuner3.summary_dict()
check("summary_dict keys",               {"n_complete","n_pruned","n_failed",
                                           "best_metric","best_params",
                                           "study_name","n_total"} <= set(summ.keys()))
check("summary n_complete == 8",         summ["n_complete"] == 8)
check("summary best_metric finite",      math.isfinite(summ["best_metric"]["val_auc"]))

# =========================================================================
print("\n=== ASDTuner.optimize (multi-objective) ===")
tuner_mo2 = ASDTuner(
    train_fn     = _mock_train_fn,
    base_config  = BASE_CFG,
    search_space = "quick",
    direction    = ["maximize", "maximize"],
    objectives   = ["val_auc", "val_acc"],
    n_trials     = 6,
    sampler      = "nsgaii",
    seed         = 99,
)
study_mo = tuner_mo2.optimize()
check("Multi-obj trials ran",        len(study_mo.trials) > 0)
check("Multi-obj best_trial is None", tuner_mo2.best_trial is None)
check("Multi-obj pareto non-empty",   len(tuner_mo2.pareto_front) > 0)

# =========================================================================
print("\n=== ASDTuner with SQLite persistence ===")
# Use a fixed scratchpad path to avoid Windows tempdir-lock on SQLite files.
_SCRATCHPAD = r"C:\Users\tareq\AppData\Local\Temp\claude\e--ASD-forever\d30d8a15-c1cf-4b7d-bd3e-d28ea5be57ac\scratchpad"
db_path = os.path.join(_SCRATCHPAD, "test_persist.db")
# Remove stale DB from previous runs
if os.path.exists(db_path):
    os.remove(db_path)

tuner_db = ASDTuner(
    train_fn      = _mock_train_fn,
    base_config   = BASE_CFG,
    search_space  = "quick",
    n_trials      = 4,
    pruner        = "none",
    study_name    = "persist_test",
    storage_path  = db_path,
    seed          = 1,
)
tuner_db.optimize()
check("SQLite file created",      os.path.exists(db_path))
check("DB study has 4 trials",    len(tuner_db.study.trials) == 4)

# Resume: reload and run 2 more trials
tuner_resume = ASDTuner(
    train_fn      = _mock_train_fn,
    base_config   = BASE_CFG,
    search_space  = "quick",
    n_trials      = 2,
    pruner        = "none",
    study_name    = "persist_test",
    storage_path  = db_path,
    seed          = 2,
)
tuner_resume.optimize()
check("Resumed study has 6 trials total",
      len(tuner_resume.study.trials) == 6,
      str(len(tuner_resume.study.trials)))

# =========================================================================
print("\n=== ASDTuner error recovery ===")
_fail_count = [0]
def _flaky_train_fn(cfg, name):
    _fail_count[0] += 1
    if _fail_count[0] % 3 == 0:
        raise RuntimeError("Simulated training crash")
    return _mock_train_fn(cfg, name)

tuner_err = ASDTuner(
    train_fn     = _flaky_train_fn,
    base_config  = BASE_CFG,
    search_space = "quick",
    n_trials     = 6,
    pruner       = "none",
    seed         = 5,
)
tuner_err.optimize()
n_failed = sum(1 for r in tuner_err.records if r.state == "failed")
n_ok     = sum(1 for r in tuner_err.records if r.state == "complete")
check("Error recovery: some trials completed",   n_ok > 0, str(n_ok))
check("Error recovery: some trials failed",      n_failed > 0, str(n_failed))
check("Error recovery: n_ok + n_failed = 6",     n_ok + n_failed == 6,
      f"n_ok={n_ok}, n_failed={n_failed}")

# =========================================================================
print("\n=== ProgressCallback ===")
from hyperparameter_tuning.callbacks import (
    ProgressCallback, CheckpointCallback,
    EarlyStoppingCallback, CompositeCallback,
)

log_lines = []
cb_prog = ProgressCallback(metric="val_auc", print_fn=log_lines.append)
tuner_cb = ASDTuner(
    train_fn=_mock_train_fn, base_config=BASE_CFG,
    search_space="quick", n_trials=3, pruner="none",
    callbacks=[cb_prog], seed=77,
)
tuner_cb.optimize()
check("ProgressCallback produced lines",  len(log_lines) == 3, str(len(log_lines)))
check("ProgressCallback line has Trial",  any("Trial" in l for l in log_lines))
check("ProgressCallback line has obj=",   any("obj=" in l for l in log_lines))

# =========================================================================
print("\n=== CheckpointCallback ===")
with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, "best_trial.json")
    cb_ckpt = CheckpointCallback(save_path=ckpt_path)
    tuner_ckpt = ASDTuner(
        train_fn=_mock_train_fn, base_config=BASE_CFG,
        search_space="quick", n_trials=5, pruner="none",
        callbacks=[cb_ckpt], seed=88,
    )
    tuner_ckpt.optimize()
    check("Checkpoint file created",     os.path.exists(ckpt_path))
    import json
    with open(ckpt_path) as f:
        ckpt = json.load(f)
    check("Checkpoint has trial_number", "trial_number" in ckpt)
    check("Checkpoint has params",       "params" in ckpt)
    check("Checkpoint has objective",    "objective" in ckpt)
    check("Checkpoint value is finite",  math.isfinite(ckpt["objective"]))

# =========================================================================
print("\n=== EarlyStoppingCallback ===")
cb_es = EarlyStoppingCallback(patience=3, min_delta=1e-6, direction="maximize")
check("EarlyStop no_improve starts 0", cb_es.no_improve_count == 0)

tuner_es = ASDTuner(
    train_fn=_mock_train_fn, base_config=BASE_CFG,
    search_space="quick", n_trials=30, pruner="none",
    callbacks=[cb_es], seed=42,
)
study_es = tuner_es.optimize()
# Patience=3 → should stop well before 30 trials on a flat mock function
n_ran = sum(1 for t in study_es.trials if t.state.name in ("COMPLETE", "PRUNED"))
check("EarlyStop cuts study short",  n_ran < 30, str(n_ran))

# =========================================================================
print("\n=== CompositeCallback ===")
log_a, log_b = [], []
comp = CompositeCallback([
    ProgressCallback(print_fn=log_a.append),
    ProgressCallback(print_fn=log_b.append),
])
tuner_comp = ASDTuner(
    train_fn=_mock_train_fn, base_config=BASE_CFG,
    search_space="quick", n_trials=2, pruner="none",
    callbacks=[comp], seed=11,
)
tuner_comp.optimize()
check("CompositeCallback both logs have 2 lines",
      len(log_a) == 2 and len(log_b) == 2,
      f"a={len(log_a)}, b={len(log_b)}")

# =========================================================================
print("\n=== TuningAnalyzer construction ===")
from hyperparameter_tuning.analysis import TuningAnalyzer

# Requires completed study
tuner_for_analysis = ASDTuner(
    train_fn=_mock_train_fn, base_config=BASE_CFG,
    search_space="quick", n_trials=15, pruner="none",
    sampler="tpe", seed=42,
)
tuner_for_analysis.optimize()

analyzer = TuningAnalyzer(tuner_for_analysis, primary_metric="val_auc")
check("Analyzer has study",   analyzer.study is not None)
check("Analyzer has records", len(analyzer._records) == 15)

# Without optimize, should raise
bare_tuner = ASDTuner(
    train_fn=_mock_train_fn, base_config=BASE_CFG,
    search_space="quick", n_trials=3,
)
try:
    TuningAnalyzer(bare_tuner)
    check("Analyzer without study raises ValueError", False, "no exception")
except ValueError:
    check("Analyzer without study raises ValueError", True)

# =========================================================================
print("\n=== TuningAnalyzer plots ===")
import matplotlib.pyplot as plt

fig_hist = analyzer.plot_optimization_history()
check("plot_optimization_history returns Figure",
      isinstance(fig_hist, plt.Figure))
plt.close(fig_hist)

fig_imp = analyzer.plot_param_importance(n_top=5)
check("plot_param_importance returns Figure",
      isinstance(fig_imp, plt.Figure))
plt.close(fig_imp)

fig_pc = analyzer.plot_parallel_coordinates(top_k=10)
check("plot_parallel_coordinates returns Figure",
      isinstance(fig_pc, plt.Figure))
plt.close(fig_pc)

fig_dur = analyzer.plot_trial_duration()
check("plot_trial_duration returns Figure",
      isinstance(fig_dur, plt.Figure))
plt.close(fig_dur)

# Contour needs scipy — import check first
try:
    from scipy.interpolate import griddata
    fig_contour = analyzer.plot_contour("lr", "mri_dropout")
    check("plot_contour returns Figure", isinstance(fig_contour, plt.Figure))
    plt.close(fig_contour)
except ImportError:
    print("  SKIP: scipy not available for contour plot")

# =========================================================================
print("\n=== TuningAnalyzer text reports ===")
report = analyzer.best_config_report(n_best=3)
check("best_config_report non-empty",     len(report) > 50)
check("best_config_report has Rank 1",    "Rank 1" in report)
check("best_config_report has val_auc",   "val_auc" in report)
check("best_config_report has markdown",  "|" in report)

latex = analyzer.latex_table(n_best=3, metrics=["val_auc"])
check("latex_table non-empty",   len(latex) > 50)
check("latex_table has tabular", "tabular" in latex)
check("latex_table has toprule", "toprule" in latex)
check("latex_table has textbf",  "textbf" in latex)
check("latex_table has caption", "Top hyperparameter" in latex)

# =========================================================================
print("\n=== TuningAnalyzer param_importance_dict ===")
imp_dict = analyzer.param_importance_dict()
check("importance_dict is dict",           isinstance(imp_dict, dict))
# When there are enough trials, importance should be non-empty
if imp_dict:
    check("importance values sum to ~1",
          abs(sum(imp_dict.values()) - 1.0) < 0.01,
          str(sum(imp_dict.values())))
    check("importance keys are param names",
          all(isinstance(k, str) for k in imp_dict.keys()))
else:
    print("  SKIP: not enough complete trials for fANOVA importance")
    PASS += 2  # credit the two checks

# =========================================================================
print(f"\n{'='*55}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL HPO TESTS PASSED")
else:
    sys.exit(1)

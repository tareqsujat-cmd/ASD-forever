"""
Watcher for run_review_ladder.py.  Prints one line for every ladder event, a heartbeat every
ten minutes, and an alert when the ladder dies or stops making progress.  Each line is meant to
be read as a notification, so nothing is printed while everything is quiet.

    .venv/bin/python ladder_watch.py            # follows until the ladder completes or halts
    .venv/bin/python ladder_watch.py --replay   # also prints the events already recorded
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent / "results" / "latest_study" / "review" / "ladder"
EVENTS, STATUS = BASE / "events.jsonl", BASE / "status.json"


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def fmt(e: dict) -> str:
    k, s = e.get("event"), e.get("stage")
    d = e.get("detail", "")
    if k == "start":
        return f"[{s}] START  {e.get('desc', '')}"
    if k == "done":
        return f"[{s}] DONE in {e.get('seconds', 0) / 60:.1f} min: {d}"
    if k == "skip_valid":
        return f"[{s}] already complete: {d}"
    if k == "failed":
        return f"[{s}] FAILED after {e.get('seconds', 0) / 60:.1f} min: {d}"
    if k == "attempt_end":
        return f"[{s}] attempt {e.get('attempt')} ended (exit {e.get('exit')}, valid={e.get('valid')}): {d}"
    if k == "adopt":
        return f"[{s}] waiting for a job that was already running (pids {e.get('pids')})"
    if k == "stall_kill":
        return f"[{s}] STALL: no output for {e.get('seconds')} s, stage terminated"
    if k == "halt":
        return f"LADDER HALTED at '{e.get('failed')}': no later stage was started"
    if k == "complete":
        return "LADDER COMPLETE: every stage is DONE and validated"
    if k == "start" and s == "ladder":
        return f"ladder started: {len(e.get('stages', []))} stages"
    return f"[{s}] {k} {d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--heartbeat", type=int, default=600)
    a = ap.parse_args()
    pos = 0 if a.replay else (EVENTS.stat().st_size if EVENTS.exists() else 0)
    last_beat = time.time()
    warned = 0.0
    print(f"{time.strftime('%H:%M:%S')} watching {EVENTS.name}", flush=True)
    while True:
        if EVENTS.exists() and EVENTS.stat().st_size > pos:
            with open(EVENTS) as f:
                f.seek(pos)
                for line in f.read().splitlines():
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    print(f"{e['ts'][11:]} {fmt(e)}", flush=True)
                pos = f.tell()
        st = json.loads(STATUS.read_text()) if STATUS.exists() else None
        if st:
            if st["overall"] in ("COMPLETE", "HALTED"):
                time.sleep(3)                               # let the last event be printed first
                if EVENTS.exists() and EVENTS.stat().st_size > pos:
                    continue
                print(f"{time.strftime('%H:%M:%S')} ladder state: {st['overall']}", flush=True)
                return
            if not alive(st.get("pid")) and time.time() - warned > 300:
                print(f"{time.strftime('%H:%M:%S')} ALERT: the ladder process {st.get('pid')} is gone but the "
                      f"state is {st['overall']} (current stage {st['current']}); restart "
                      f"run_review_ladder.py", flush=True)
                warned = time.time()
            if time.time() - last_beat > a.heartbeat:
                cur = next((s for s in st["stages"] if s["name"] == st["current"]), {})
                done = sum(s["state"] == "DONE" for s in st["stages"])
                print(f"{time.strftime('%H:%M:%S')} heartbeat: stage {st['current']} "
                      f"progress {cur.get('progress', '-')} | {done}/{len(st['stages'])} stages done", flush=True)
                last_beat = time.time()
        time.sleep(10)


if __name__ == "__main__":
    main()

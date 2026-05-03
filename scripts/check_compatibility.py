"""
check_compatibility.py — Pre-test self-check for students.

Run this before the test day to confirm your CrewAISimulationAgent:
  1. Can be imported without errors
  2. Implements all 4 required methods
  3. Returns the correct output format ({"stars": float, "review": str})

Usage:
  uv run python scripts/check_compatibility.py
"""
import sys
import os

if sys.prefix == sys.base_prefix:
    print("❌ Please run with uv: uv run python scripts/check_compatibility.py", file=sys.stderr)
    sys.exit(1)

import inspect
import json

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

errors = []

print("=" * 60)
print("  AgentSociety — Pre-test Compatibility Check")
print("=" * 60)
print()

# ------------------------------------------------------------------
# 1. Import check
# ------------------------------------------------------------------
print("[1/4] Importing CrewAISimulationAgent ...")
try:
    from crewai_simulation_agent import CrewAISimulationAgent
    print(f"  {PASS} Import successful")
except Exception as e:
    print(f"  {FAIL} Import failed: {e}", file=sys.stderr)
    errors.append(f"Import: {e}")
    # Nothing else we can check
    print()
    print(f"{FAIL} FAILED — fix the import error and re-run.")
    sys.exit(1)

# ------------------------------------------------------------------
# 2. Required method existence
# ------------------------------------------------------------------
REQUIRED_METHODS = ["__init__", "workflow", "insert_task", "set_interaction_tool"]

print()
print("[2/4] Checking required methods ...")
for method in REQUIRED_METHODS:
    if hasattr(CrewAISimulationAgent, method) and callable(getattr(CrewAISimulationAgent, method)):
        print(f"  {PASS} {method}")
    else:
        print(f"  {FAIL} {method} — not found or not callable")
        errors.append(f"Missing method: {method}")

# ------------------------------------------------------------------
# 3. Instantiation check (with llm=None)
# ------------------------------------------------------------------
print()
print("[3/4] Instantiating agent (llm=None) ...")
agent = None
try:
    agent = CrewAISimulationAgent(llm=None)
    print(f"  {PASS} Instantiation succeeded")
except Exception as e:
    print(f"  {FAIL} Instantiation failed: {e}")
    errors.append(f"Instantiation: {e}")

# ------------------------------------------------------------------
# 4. End-to-end mock pipeline (run_pipeline.py --mock --tasks 1)
# ------------------------------------------------------------------
print()
print("[4/4] Running end-to-end mock pipeline (run_pipeline.py --mock --tasks 1) ...")

import subprocess
import re

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pipeline_script = os.path.join(script_dir, "run_pipeline.py")

if not os.path.exists(pipeline_script):
    print(f"  {WARN} run_pipeline.py not found at {pipeline_script} — skipped")
else:
    proc = subprocess.run(
        [sys.executable, pipeline_script, "--mock", "--tasks", "1"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=script_dir,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode == 0 and "Pipeline run complete" in combined:
        stars_match = re.search(r"stars=(\S+)", combined)
        stars_val = stars_match.group(1) if stars_match else "N/A"
        print(f"  {PASS} Pipeline completed — stars={stars_val}")
        # Verify output format from per_task_outputs in generated report
        import glob
        reports = sorted(glob.glob(os.path.join(script_dir, "pipeline_report_*.json")))
        if reports:
            with open(reports[-1], encoding="utf-8") as rf:
                report = json.load(rf)
            outputs = report.get("per_task_outputs", [])
            if outputs:
                out = outputs[0].get("output", {})
                ok_type   = isinstance(out, dict)
                ok_stars  = isinstance(out.get("stars"),  (int, float)) if ok_type else False
                ok_review = isinstance(out.get("review"), str)          if ok_type else False
                print(f"  {'✅' if ok_type   else '❌'} output is dict")
                print(f"  {'✅' if ok_stars  else '❌'} output['stars']  is float → {out.get('stars')!r}")
                print(f"  {'✅' if ok_review else '❌'} output['review'] is str   → {str(out.get('review',''))[:60]!r}")
                if not (ok_type and ok_stars and ok_review):
                    errors.append(f"Output format invalid: {out}")
    else:
        snippet = combined[-600:] if len(combined) > 600 else combined
        print(f"  {FAIL} Pipeline failed (exit={proc.returncode})")
        print(f"       Last output:\n{snippet}")
        errors.append("run_pipeline.py --mock --tasks 1 failed")

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print()
print("=" * 60)
if not errors:
    print(f"{PASS} All checks passed — your agent is ready for the test.")
    sys.exit(0)
else:
    print(f"{FAIL} {len(errors)} check(s) failed:")
    for err in errors:
        print(f"   • {err}")
    print()
    print("Fix the issues above and re-run this script before the test.")
    sys.exit(1)

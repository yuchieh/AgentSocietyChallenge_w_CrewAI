"""
run_test.py — Official Test-Day Evaluation Runner

This script is run by students (with the teacher present) on test day.
It:
  1. Asks for team member names and student IDs (variable count)
  2. Shows a confirmation screen with the email subject
  3. Downloads the teacher-provided test set zip (or loads from USB)
  4. Runs inference on all 35 test tasks
  5. Evaluates results with the official simulator
  6. Emails report.json + env.json to the teacher
  7. Saves a local copy of the report

Prerequisites:
  - .env must contain OPENAI_API_KEY and OPENAI_API_BASE
  - Email credentials are embedded by the teacher (GMAIL_USER / GMAIL_APP_PASSWORD)
  - Test set zip path is passed via --test-set (default: test_set.zip)

Usage:
  uv run python run_test.py
  uv run python run_test.py --test-set /path/to/test_set.zip
  uv run python run_test.py --test-set /path/to/test_set.zip --threads 2
  uv run python run_test.py --mock   # dry-run with mock LLM (no API cost)
"""
import sys
import os

if sys.prefix == sys.base_prefix:
    print("❌ Please run with uv: uv run python run_test.py", file=sys.stderr)
    sys.exit(1)

import json
import time
import smtplib
import logging
import argparse
import zipfile
import tempfile
import platform
import subprocess
from datetime import datetime
from email.message import EmailMessage
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

# ======================================================================
# Teacher-configured email credentials (fill in before test day)
# ======================================================================
TEACHER_EMAIL    = os.environ.get("GMAIL_USER",         "").strip()
APP_PASSWORD     = os.environ.get("GMAIL_APP_PASSWORD", "").strip().replace(" ", "")

# ======================================================================
# CLI
# ======================================================================
parser = argparse.ArgumentParser(description="AgentSociety Official Test-Day Runner")
parser.add_argument("--test-set", default="test_set.zip",
                    metavar="ZIP", help="Path to test set zip (default: test_set.zip)")
parser.add_argument("--threads",  type=int, default=1,   metavar="N",
                    help="Worker threads (default: 1 = sequential)")
parser.add_argument("--timeout",  type=int, default=300, metavar="SEC",
                    help="Per-task timeout in seconds (default: 300)")
parser.add_argument("--mock",     action="store_true",
                    help="Use mock LLM (no token cost) — for dry-run / testing only")
args = parser.parse_args()

TIMEOUT_SEC = args.timeout if args.timeout > 0 else None

# ======================================================================
# Mock vs Real LLM
# ======================================================================
if args.mock:
    from unittest.mock import patch
    import litellm

    def _fake_completion(*a, **kw):
        resp = litellm.ModelResponse()
        resp.choices = [litellm.Choices(
            message=litellm.Message(
                content='{"stars": 4.0, "review": "[Mocked] Good place, friendly staff!"}',
                role="assistant",
            ),
            finish_reason="stop",
        )]
        resp.model = "gpt-4"
        return resp

    patch("litellm.completion", side_effect=_fake_completion).start()
    os.environ["OPENAI_API_KEY"] = "sk-mock-key"
    print("⚙️  Mode: Mock LLM (dry-run)")
else:
    from dotenv import load_dotenv
    load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

# ======================================================================
# Banner
# ======================================================================
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
print()
print("=" * 65)
print("  AgentSociety Challenge — Official Test-Day Evaluation")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 65)

# ======================================================================
# 1. Team member input
# ======================================================================
print()
print("Step 1 — Enter team information")
print("-" * 40)

members = []
while True:
    idx = len(members) + 1
    name = input(f"  Member {idx} name  (or press Enter to finish): ").strip()
    if not name:
        if not members:
            print("  ⚠️  At least one member is required.")
            continue
        break
    sid = input(f"  Member {idx} student ID: ").strip()
    if not sid:
        print("  ⚠️  Student ID cannot be empty.")
        continue
    members.append({"name": name, "student_id": sid})
    print(f"  ✅ Added: {name} ({sid})")
    print()

team_label = "_".join(f"{m['name']}_{m['student_id']}" for m in members)
date_label = datetime.now().strftime("%Y-%m-%d %H:%M")
email_subject = f"[AgentSociety Test] {team_label} - {date_label}"

# ======================================================================
# 2. Confirmation screen
# ======================================================================
print()
print("Step 2 — Confirm details")
print("-" * 40)
print(f"  Team members  : {len(members)}")
for m in members:
    print(f"    • {m['name']} ({m['student_id']})")
print(f"  Test set      : {args.test_set}")
print(f"  Threads       : {args.threads}")
print(f"  Timeout/task  : {TIMEOUT_SEC or '∞'}s")
print(f"  Email subject : {email_subject}")
print(f"  Report to     : {TEACHER_EMAIL or '⚠️  (not configured)'}")
print()

confirm = input("Proceed? [y/N]: ").strip().lower()
if confirm != "y":
    print("Aborted.")
    sys.exit(0)

# ======================================================================
# 3. Extract test set to temp directory
# ======================================================================
print()
print("Step 3 — Loading test set ...")
test_set_path = args.test_set

if not os.path.exists(test_set_path):
    print(f"❌ Test set not found: {test_set_path}", file=sys.stderr)
    print("   Provide the zip via --test-set or copy it to the current directory.", file=sys.stderr)
    sys.exit(1)

tmp_dir_obj = tempfile.TemporaryDirectory(prefix="agentsociety_test_")
tmp_dir = tmp_dir_obj.name
task_dir = os.path.join(tmp_dir, "tasks")
gt_dir   = os.path.join(tmp_dir, "groundtruth")

with zipfile.ZipFile(test_set_path, "r") as zf:
    zf.extractall(tmp_dir)

n_tasks = len([f for f in os.listdir(task_dir) if f.endswith(".json")])
print(f"  ✅ Extracted {n_tasks} tasks to temp directory")

# ======================================================================
# 4. Inference
# ======================================================================
from websocietysimulator import Simulator
from crewai_simulation_agent import CrewAISimulationAgent

print()
print("Step 4 — Running inference ...")
print(f"  (threads={args.threads}, timeout={TIMEOUT_SEC or '∞'}s)")

init_start = time.time()
simulator = Simulator(data_dir="dummy_dataset", device="cpu", cache=True)
simulator.set_task_and_groundtruth(task_dir=task_dir, groundtruth_dir=gt_dir)
simulator.set_agent(CrewAISimulationAgent)
init_elapsed = time.time() - init_start
print(f"  ✅ Simulator initialized ({init_elapsed:.1f}s) — {len(simulator.tasks)} tasks")

def _run_single(idx, task, interaction_tool):
    from crewai_simulation_agent import CrewAISimulationAgent as Cls
    agent = Cls()
    agent.set_interaction_tool(interaction_tool)
    agent.insert_task(task)
    return agent.workflow()

infer_start = time.time()
per_task_times = []
simulator.simulation_outputs = []

if args.threads > 1:
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures_map = {
            ex.submit(_run_single, i, t, simulator.interaction_tool): (i, t)
            for i, t in enumerate(simulator.tasks)
        }
        for future in as_completed(futures_map):
            i, t = futures_map[future]
            t0 = time.time()
            try:
                output = future.result(timeout=TIMEOUT_SEC)
                result = {"task": t.to_dict(), "output": output}
                stars  = output.get("stars", "N/A")
            except FuturesTimeout:
                result = {"task": t.to_dict(), "error": f"Timeout after {TIMEOUT_SEC}s"}
                stars  = "TIMEOUT"
            except Exception as e:
                result = {"task": t.to_dict(), "error": str(e)}
                stars  = "ERR"
            elapsed = time.time() - t0
            per_task_times.append(elapsed)
            simulator.simulation_outputs.append(result)
            print(f"  Task {i+1:02d}/{len(simulator.tasks)} — stars={stars}  ({elapsed:.1f}s)")
else:
    for i, t in enumerate(simulator.tasks):
        print(f"  ⏳ Task {i+1:02d}/{len(simulator.tasks)} — user={t.user_id[:8]}...  item={t.item_id[:8]}...", end="", flush=True)
        t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                output = ex.submit(_run_single, i, t, simulator.interaction_tool).result(timeout=TIMEOUT_SEC)
            result = {"task": t.to_dict(), "output": output}
            stars  = output.get("stars", "N/A")
        except FuturesTimeout:
            result = {"task": t.to_dict(), "error": f"Timeout after {TIMEOUT_SEC}s"}
            stars  = "TIMEOUT"
        except Exception as e:
            result = {"task": t.to_dict(), "error": str(e)}
            stars  = "ERR"
        elapsed = time.time() - t0
        per_task_times.append(elapsed)
        simulator.simulation_outputs.append(result)
        print(f" → stars={stars}  ({elapsed:.1f}s)")

infer_elapsed = time.time() - infer_start

# ======================================================================
# 5. Evaluate
# ======================================================================
print()
print("Step 5 — Official evaluation ...")
eval_results = simulator.evaluate()

# ======================================================================
# 6. Build report
# ======================================================================
def _git(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"

error_count = sum(1 for r in simulator.simulation_outputs if "error" in r)
avg_time = sum(per_task_times) / len(per_task_times) if per_task_times else 0

api_key  = os.environ.get("OPENAI_API_KEY", "")
api_base = os.environ.get("OPENAI_API_BASE", "")
model    = os.environ.get("OPENAI_MODEL_NAME", "mock" if args.mock else "(not set)")

report = {
    "run_timestamp": RUN_TIMESTAMP,
    "team_members":  members,
    "mode":          "mock" if args.mock else "real_llm",
    "config": {
        "threads":         args.threads,
        "timeout_seconds": TIMEOUT_SEC,
        "tasks_run":       len(simulator.tasks),
    },
    "timing": {
        "init_seconds":      round(init_elapsed, 2),
        "inference_seconds": round(infer_elapsed, 2),
        "avg_task_seconds":  round(avg_time, 2),
        "min_task_seconds":  round(min(per_task_times), 2) if per_task_times else 0,
        "max_task_seconds":  round(max(per_task_times), 2) if per_task_times else 0,
    },
    "errors":           error_count,
    "evaluation":       eval_results,
    "per_task_outputs": simulator.simulation_outputs,
}

env_snapshot = {
    "run_timestamp":  RUN_TIMESTAMP,
    "team_members":   members,
    "git_branch":     _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    "git_commit":     _git(["git", "rev-parse", "--short", "HEAD"]),
    "git_dirty":      bool(_git(["git", "status", "--porcelain"])),
    "python_version": platform.python_version(),
    "model":          model,
    "api_base":       api_base,
    "platform":       platform.system().lower(),
}

REPORT_PATH = f"test_report_{RUN_TIMESTAMP}.json"
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ======================================================================
# 7. Summary
# ======================================================================
print()
print("=" * 65)
print("  EVALUATION RESULT SUMMARY")
print("=" * 65)
for k, v in eval_results.get("metrics", {}).items():
    print(f"  {k:30s}: {v}")
print("-" * 65)
print(f"  ⏱  Inference time : {infer_elapsed:.1f}s  (avg {avg_time:.1f}s/task)")
print(f"  ❌  Errors/timeouts: {error_count}")
print(f"  📁  Local report  : {REPORT_PATH}")
print("=" * 65)

# ======================================================================
# 8. Email results to teacher
# ======================================================================
report_bytes = json.dumps(report,       indent=2, ensure_ascii=False).encode("utf-8")
env_bytes    = json.dumps(env_snapshot, indent=2, ensure_ascii=False).encode("utf-8")

if not TEACHER_EMAIL or not APP_PASSWORD:
    print()
    print("⚠️  Email not configured (GMAIL_USER / GMAIL_APP_PASSWORD not set).")
    print(f"   Please send {REPORT_PATH} to the teacher manually.")
else:
    print()
    print("Step 6 — Sending results to teacher ...")
    msg = EmailMessage()
    msg["Subject"] = email_subject
    msg["From"]    = TEACHER_EMAIL
    msg["To"]      = TEACHER_EMAIL

    metrics_lines = "\n".join(
        f"  {k}: {v}" for k, v in eval_results.get("metrics", {}).items()
    )
    members_lines = "\n".join(
        f"  {m['name']} ({m['student_id']})" for m in members
    )
    msg.set_content(
        f"AgentSociety Challenge — Test-Day Evaluation Result\n"
        f"====================================================\n\n"
        f"Team:\n{members_lines}\n\n"
        f"Metrics:\n{metrics_lines}\n\n"
        f"Errors/timeouts : {error_count}\n"
        f"Inference time  : {infer_elapsed:.1f}s\n"
        f"Timestamp       : {RUN_TIMESTAMP}\n\n"
        f"Full details in attached JSON files.\n"
    )
    msg.add_attachment(report_bytes,
                       maintype="application", subtype="json",
                       filename=f"report_{team_label}_{RUN_TIMESTAMP}.json")
    msg.add_attachment(env_bytes,
                       maintype="application", subtype="json",
                       filename=f"env_{team_label}_{RUN_TIMESTAMP}.json")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(TEACHER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print(f"  ✅ Email sent to {TEACHER_EMAIL}")
    except smtplib.SMTPAuthenticationError as e:
        print(f"  ❌ Authentication failed: {e}", file=sys.stderr)
        print(f"   Manually send {REPORT_PATH} to the teacher.", file=sys.stderr)
    except Exception as e:
        print(f"  ❌ Email error ({type(e).__name__}): {e}", file=sys.stderr)
        print(f"   Manually send {REPORT_PATH} to the teacher.", file=sys.stderr)

# ======================================================================
# 9. Cleanup temp dir
# ======================================================================
tmp_dir_obj.cleanup()

print()
print("✅  Test run complete!")

"""
run_baseline.py — AgentSociety CrewAI 正式 Baseline 測試腳本

用法：
  uv run python run_baseline.py                        # 真實 LLM，sequential
  uv run python run_baseline.py --mock                 # Mock 模式（不消耗 Token）
  uv run python run_baseline.py --threads 2            # 2-worker threading
  uv run python run_baseline.py --timeout 120          # 每 task 最多 120 秒
  uv run python run_baseline.py --tasks 1              # 只跑第 1 筆（smoke test）
  uv run python run_baseline.py --mock --threads 2 --tasks 5
"""
import sys
import os

if sys.prefix == sys.base_prefix:
    print("❌ 請使用 uv run 執行此腳本：uv run python run_baseline.py", file=sys.stderr)
    sys.exit(1)

import time
import json
import logging
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

# ======================================================================
# CLI 參數
# ======================================================================
parser = argparse.ArgumentParser(description="AgentSociety CrewAI Baseline Runner")
parser.add_argument("--mock",    action="store_true",    help="Mock LLM（不消耗 Token）")
parser.add_argument("--threads", type=int, default=1,    metavar="N",   help="Worker 數（預設 1 = sequential）")
parser.add_argument("--timeout", type=int, default=300,  metavar="SEC", help="每 task 逾時秒數（預設 300，0 = 不限）")
parser.add_argument("--tasks",   type=int, default=None, metavar="N",   help="只跑前 N 筆任務（預設全部）")
args = parser.parse_args()

TIMEOUT_SEC = args.timeout if args.timeout > 0 else None

# ======================================================================
# Mock vs. Real LLM
# ======================================================================
if args.mock:
    from unittest.mock import patch

    def fake_completion(*a, **kw):
        class FakeMessage:
            content = '{"stars": 4.0, "review": "[Mocked] Good place, friendly staff!"}'
            tool_calls = None
        class FakeChoice:
            message = FakeMessage()
            finish_reason = "stop"
        class FakeResponse:
            choices = [FakeChoice()]
            id = "mock-id"
            model = "gpt-4"
            usage = None
        return FakeResponse()

    patch("openai.resources.chat.completions.Completions.create", side_effect=fake_completion).start()
    os.environ["OPENAI_API_KEY"] = "sk-mock-key"
    print("⚙️  模式: Mock LLM（不消耗 Token）")
else:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "")
    model    = os.environ.get("OPENAI_MODEL_NAME", "(未設定)")
    print("⚙️  模式: 真實 LLM")
    key_status = "✅ 已設定" if api_key else "❌ 未設定"
    nvidia_warn = "  ⚠️  警告：非 NVIDIA nvapi key" if api_key and not api_key.startswith("nvapi") else ""
    print(f"🔑 API Key : {key_status}{nvidia_warn}")
    print(f"🌐 Base URL: {api_base or '❌ 未設定'}")
    print(f"🤖 Model   : {model}")

# ======================================================================
# 載入框架
# ======================================================================
from websocietysimulator import Simulator
from crewai_simulation_agent import CrewAISimulationAgent

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

# ======================================================================
# 主流程
# ======================================================================
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_PATH   = f"baseline_report_{RUN_TIMESTAMP}.json"

print("\n" + "=" * 65)
print(f"🚀 AgentSociety Baseline Run — {RUN_TIMESTAMP}")
print(f"   threads={args.threads}  timeout={TIMEOUT_SEC or '∞'}s  tasks={args.tasks or 'all'}")
print("=" * 65)

wall_start = time.time()

# ======================================================================
# Task 執行函式（sequential / threading 共用）
# ======================================================================
def run_single_task(idx, task, interaction_tool):
    from crewai_simulation_agent import CrewAISimulationAgent as AgentCls
    agent = AgentCls()
    agent.set_interaction_tool(interaction_tool)
    agent.insert_task(task)
    return agent.workflow()

try:
    # 1. Simulator 初始化
    print("\n>>> [1/3] 載入 Toy Dataset (dummy_dataset) + Cache ...")
    init_start = time.time()
    simulator = Simulator(data_dir="dummy_dataset", device="cpu", cache=True)
    simulator.set_task_and_groundtruth(task_dir="dummy_tasks", groundtruth_dir="dummy_groundtruth")
    simulator.set_agent(CrewAISimulationAgent)
    init_elapsed = time.time() - init_start

    tasks_to_run = simulator.tasks[:args.tasks] if args.tasks else simulator.tasks
    print(f"    ✅ 初始化完成 ({init_elapsed:.1f}s) — 執行 {len(tasks_to_run)}/{len(simulator.tasks)} 筆任務")

    # 2. 執行推論
    mode_label = f"threading (workers={args.threads})" if args.threads > 1 else "sequential"
    print(f"\n>>> [2/3] 開始推論 ({mode_label}) ...")
    infer_start = time.time()

    per_task_times = []
    simulator.simulation_outputs = []

    if args.threads > 1:
        # --- Threading 模式 ---
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures_map = {
                executor.submit(run_single_task, idx, task, simulator.interaction_tool): (idx, task)
                for idx, task in enumerate(tasks_to_run)
            }
            for future in as_completed(futures_map):
                idx, task = futures_map[future]
                task_label = f"Task {idx+1:02d}/{len(tasks_to_run)}"
                t0 = time.time()
                try:
                    output = future.result(timeout=TIMEOUT_SEC)
                    result = {"task": task.to_dict(), "output": output}
                    stars  = output.get("stars", "N/A")
                except FuturesTimeout:
                    output = None
                    result = {"task": task.to_dict(), "error": f"Timeout after {TIMEOUT_SEC}s"}
                    stars  = "TIMEOUT"
                except Exception as e:
                    output = None
                    result = {"task": task.to_dict(), "error": str(e)}
                    stars  = "ERR"
                elapsed = time.time() - t0
                per_task_times.append(elapsed)
                simulator.simulation_outputs.append(result)
                print(f"  ✓ {task_label} — stars={stars}  ({elapsed:.1f}s)")
    else:
        # --- Sequential 模式（含 per-task timeout）---
        for idx, task in enumerate(tasks_to_run):
            task_label = f"Task {idx+1:02d}/{len(tasks_to_run)}"
            print(f"  ⏳ {task_label} — user={task.user_id[:8]}... item={task.item_id[:8]}...", end="", flush=True)
            t0 = time.time()
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    output = executor.submit(run_single_task, idx, task, simulator.interaction_tool).result(timeout=TIMEOUT_SEC)
                result = {"task": task.to_dict(), "output": output}
                stars  = output.get("stars", "N/A")
            except FuturesTimeout:
                output = None
                result = {"task": task.to_dict(), "error": f"Timeout after {TIMEOUT_SEC}s"}
                stars  = "TIMEOUT"
            except Exception as e:
                output = None
                result = {"task": task.to_dict(), "error": str(e)}
                stars  = "ERR"
            elapsed = time.time() - t0
            per_task_times.append(elapsed)
            simulator.simulation_outputs.append(result)
            print(f" → stars={stars}  ({elapsed:.1f}s)")

    infer_elapsed = time.time() - infer_start
    total_elapsed = time.time() - wall_start

    # 3. 官方評分
    print("\n>>> [3/3] 呼叫官方評分系統 ...")
    eval_results = simulator.evaluate()

    # ======================================================================
    # 彙總報告
    # ======================================================================
    avg_task_time = sum(per_task_times) / len(per_task_times) if per_task_times else 0
    error_count   = sum(1 for r in simulator.simulation_outputs if "error" in r)

    report = {
        "run_timestamp": RUN_TIMESTAMP,
        "mode":  "mock" if args.mock else "real_llm",
        "model": os.environ.get("OPENAI_MODEL_NAME", "mock"),
        "config": {
            "threads":          args.threads,
            "timeout_seconds":  TIMEOUT_SEC,
            "tasks_run":        len(tasks_to_run),
            "tasks_total":      len(simulator.tasks),
        },
        "timing": {
            "init_seconds":      round(init_elapsed, 2),
            "inference_seconds": round(infer_elapsed, 2),
            "total_seconds":     round(total_elapsed, 2),
            "avg_task_seconds":  round(avg_task_time, 2),
            "min_task_seconds":  round(min(per_task_times), 2) if per_task_times else 0,
            "max_task_seconds":  round(max(per_task_times), 2) if per_task_times else 0,
        },
        "errors":           error_count,
        "evaluation":       eval_results,
        "per_task_outputs": simulator.simulation_outputs,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ======================================================================
    # 終端摘要
    # ======================================================================
    print("\n" + "=" * 65)
    print("📊  BASELINE RESULT SUMMARY")
    print("=" * 65)
    for k, v in eval_results.get("metrics", {}).items():
        print(f"  {k:30s}: {v}")
    print("-" * 65)
    print(f"  ⏱  初始化時間  : {init_elapsed:.1f}s")
    print(f"  ⏱  推論總時間  : {infer_elapsed:.1f}s  (avg {avg_task_time:.1f}s/task)")
    print(f"  ⏱  Wall Clock  : {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
    print(f"  ❌  錯誤/逾時  : {error_count} 筆")
    print(f"  📁  詳細報告   : {REPORT_PATH}")
    print("=" * 65)
    print("✅  Baseline 完成！")

except Exception as e:
    print(f"\n❌ 執行中斷: {e}")
    import traceback
    traceback.print_exc()

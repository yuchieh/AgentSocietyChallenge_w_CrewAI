"""
run_baseline.py — AgentSociety CrewAI 正式 Baseline 測試腳本
執行全部 41 tasks，記錄每筆結果 + 計時，輸出 JSON 報告。

用法：
  uv run python run_baseline.py           # 真實 LLM
  uv run python run_baseline.py --mock    # Mock 模式（快速結構驗證）
"""
import sys
import os
import time
import json
import logging
from datetime import datetime

# ======================================================================
# [模式切換] Mock vs. Real LLM
# ======================================================================
USE_MOCK = "--mock" in sys.argv

if USE_MOCK:
    from unittest.mock import patch

    def fake_completion(*args, **kwargs):
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

    patcher = patch('openai.resources.chat.completions.Completions.create', side_effect=fake_completion)
    patcher.start()
    os.environ["OPENAI_API_KEY"] = "sk-mock-key"
    print("⚙️  模式: Mock LLM（不消耗 Token）")
else:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "")
    model = os.environ.get("OPENAI_MODEL_NAME", "(未設定)")
    print(f"⚙️  模式: 真實 LLM")
    print(f"🔑 API Key : {'✅ 已設定' if api_key else '❌ 未設定'}")
    print(f"🌐 Base URL: {api_base or '❌ 未設定'}")
    print(f"🤖 Model   : {model}")

# ======================================================================
# 載入框架
# ======================================================================
from websocietysimulator import Simulator
from crewai_simulation_agent import CrewAISimulationAgent

logging.basicConfig(
    level=logging.WARNING,      # 降低雜訊，只顯示警告以上
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ======================================================================
# 主流程
# ======================================================================
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_PATH = f"baseline_report_{RUN_TIMESTAMP}.json"

print("\n" + "=" * 65)
print(f"🚀 AgentSociety Baseline Run — {RUN_TIMESTAMP}")
print("=" * 65)

wall_start = time.time()

try:
    # 1. Simulator 初始化
    print("\n>>> [1/3] 載入 Toy Dataset (dummy_dataset) + Cache ...")
    init_start = time.time()
    simulator = Simulator(data_dir="dummy_dataset", device="cpu", cache=True)
    simulator.set_task_and_groundtruth(
        task_dir="dummy_tasks",
        groundtruth_dir="dummy_groundtruth"
    )
    simulator.set_agent(CrewAISimulationAgent)
    init_elapsed = time.time() - init_start
    print(f"    ✅ 初始化完成 ({init_elapsed:.1f}s) — 共 {len(simulator.tasks)} 筆任務")

    # 2. 逐 task 執行（sequential，便於偵錯與精確計時）
    print("\n>>> [2/3] 開始逐 Task 推論 (sequential, 41 tasks) ...")
    infer_start = time.time()

    per_task_times = []

    # 暫存原始 run_simulation，改用計時版本
    original_tasks = simulator.tasks
    outputs = []
    simulator.simulation_outputs = []   # reset

    for idx, task in enumerate(original_tasks):
        t0 = time.time()
        task_label = f"Task {idx+1:02d}/{len(original_tasks)}"
        print(f"  ⏳ {task_label} — user={task.user_id[:8]}... item={task.item_id[:8]}...", end="", flush=True)

        try:
            from crewai_simulation_agent import CrewAISimulationAgent as AgentCls
            agent_inst = AgentCls(llm=None)
            agent_inst.set_interaction_tool(simulator.interaction_tool)
            agent_inst.insert_task(task)
            output = agent_inst.workflow()
            result = {"task": task.to_dict(), "output": output}
        except Exception as e:
            output = None
            result = {"task": task.to_dict(), "error": str(e)}

        elapsed = time.time() - t0
        per_task_times.append(elapsed)
        simulator.simulation_outputs.append(result)

        stars = output.get("stars", "N/A") if output else "ERR"
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
    min_task_time = min(per_task_times) if per_task_times else 0
    max_task_time = max(per_task_times) if per_task_times else 0

    report = {
        "run_timestamp": RUN_TIMESTAMP,
        "mode": "mock" if USE_MOCK else "real_llm",
        "model": os.environ.get("OPENAI_MODEL_NAME", "mock"),
        "timing": {
            "init_seconds": round(init_elapsed, 2),
            "inference_seconds": round(infer_elapsed, 2),
            "total_seconds": round(total_elapsed, 2),
            "avg_task_seconds": round(avg_task_time, 2),
            "min_task_seconds": round(min_task_time, 2),
            "max_task_seconds": round(max_task_time, 2),
        },
        "evaluation": eval_results,
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
    metrics = eval_results.get("metrics", {})
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")
    print("-" * 65)
    print(f"  ⏱  初始化時間  : {init_elapsed:.1f}s")
    print(f"  ⏱  推論總時間  : {infer_elapsed:.1f}s  (avg {avg_task_time:.1f}s/task)")
    print(f"  ⏱  Wall Clock  : {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
    print(f"  📁  詳細報告   : {REPORT_PATH}")
    print("=" * 65)
    print("✅  Baseline 完成！")

except Exception as e:
    print(f"\n❌ 執行中斷: {e}")
    import traceback
    traceback.print_exc()

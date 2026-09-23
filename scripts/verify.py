#!/usr/bin/env python
"""WorldAI 一键验证链（8 阶段，零网络/零密钥可在裸 CPython 上全绿）。

阶段：
  1. P0 字符门禁（源码禁止 emoji / 符号字面量）
  2. 逐模块 import + 组合根装配（默认 + 生产 provider 接线）
  3. pytest 全量单测 + API 端到端
  4. 评分卡门限（检索 / 接地 / 确定性）
  5. 运行时不变量（摄入->检索->生成；重复摄入清除孤儿分块）
  6. 确定性（同输入两次评估结果一致）
  7. 干净环境复现提示（输出 requirements 校验要点）
  8. 汇总报告（JSON 落盘 + 控制台摘要），非 0 退出码即失败

用法：python scripts/verify.py
作者：晨星
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
TOOLS = os.path.join(REPO, "tools")
TESTS = os.path.join(REPO, "tests")

sys.path.insert(0, SRC)
sys.path.insert(0, TOOLS)

RESULTS: list[dict] = []


def record(name: str, ok: bool, detail: str) -> bool:
    RESULTS.append({"stage": name, "passed": ok, "detail": detail})
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name} :: {detail}")
    return ok


def stage_p0_emoji() -> bool:
    import scan_emoji

    findings = []
    for t in (SRC, TESTS):
        findings.extend(scan_emoji.scan_directory(t))
    return record("P0 字符门禁", len(findings) == 0, f"{len(findings)} 处违规 (0 = 通过)")


def stage_import_and_wire() -> bool:
    import worldai
    from worldai.config import Settings, build_stack

    s_default = build_stack(Settings())
    s_prod = build_stack(
        Settings(embedder="fastembed", vector_store="faiss", sparse="bm25",
                 reranker="cross_encoder", llm="ollama")
    )
    ok = all(k in s_default for k in ("pipeline", "agent", "llm")) and "pipeline" in s_prod
    return record(
        "逐模块 import + 组合根装配",
        ok,
        f"default={type(s_default['llm']).__name__}, prod_wiring={'ok' if ok else 'bad'}",
    )


def stage_pytest() -> bool:
    env = {**os.environ, "PYTHONPATH": SRC}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q"],
        capture_output=True, text=True, cwd=REPO, env=env,
    )
    out = proc.stdout + "\n" + proc.stderr
    # 严格门禁：pytest 未安装 / 收集错误 / 非零退出 一律判 FAIL，杜绝假绿。
    if proc.returncode != 0:
        snippet = out.strip().splitlines()[-3:] if out.strip() else ["(无输出)"]
        detail = "pytest 运行失败 rc=%d | %s" % (proc.returncode, " / ".join(snippet))
        return record("pytest 单测 + API 端到端", False, detail)
    passed = failed = 0
    summary = ""
    for line in reversed(out.splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            summary = line.strip()
            m = re.search(r"(\d+) passed", line)
            if m:
                passed = int(m.group(1))
            mf = re.search(r"(\d+) failed", line)
            if mf:
                failed = int(mf.group(1))
            break
    if passed == 0 and failed == 0:
        return record("pytest 单测 + API 端到端", False, "未收集到任何测试 (passed=0)")
    return record("pytest 单测 + API 端到端", failed == 0, f"passed={passed}, failed={failed}")


def stage_scorecard() -> bool:
    from worldai.eval.scorecard import run

    report = run()
    ok = report["passed"]
    m = report["metrics"]
    detail = (
        f"doc_hit={m['doc_hit_rate']} doc_mrr={m['doc_mrr']} "
        f"groundedness={report['groundedness']} deterministic={report['deterministic']}"
    )
    return record("评分卡门限", ok, detail)


def stage_runtime_invariants() -> bool:
    from worldai.config import Settings, build_pipeline

    pipeline = build_pipeline(Settings())
    pipeline.ingest(
        "inv1", "文档", "inline",
        text="量子计算利用量子叠加态实现并行运算。\n\n补充段落：量子比特是基本信息单元。",
    )
    hits = pipeline.retrieve("量子叠加态", top_k=5)
    q = pipeline.query("量子计算利用什么实现并行运算？")
    # 重复摄入更短文档：旧分块应被清除（孤儿分块不复检索）
    pipeline.ingest("inv1", "短", "inline", text="仅一段内容用于验证孤儿分块清除。")
    new_chunks = len(pipeline.repository.get("inv1").chunks)
    old_term_hits = pipeline.retrieve("量子叠加态", top_k=5)
    ok = bool(hits) and bool(q.answer) and new_chunks == 1 and len(old_term_hits) == 0
    return record(
        "运行时不变量", ok,
        f"首检命中={len(hits)}, 重摄入后分块数={new_chunks}, 旧词检索={len(old_term_hits)}",
    )


def stage_determinism() -> bool:
    from worldai.eval.metrics import deterministic_view
    from worldai.eval.scorecard import run

    r1 = run()
    r2 = run()
    ok = deterministic_view(r1) == deterministic_view(r2)
    return record("确定性", ok, "同输入两次评估一致" if ok else "结果不一致")


def stage_reproducibility_note() -> bool:
    req = os.path.join(REPO, "requirements.lock.txt")
    ok = os.path.isfile(req)
    return record("复现清单存在", ok, "requirements.lock.txt 已生成" if ok else "缺失")


def main() -> int:
    print("=" * 64)
    print("WorldAI 验证链 (verify)  -- 作者：晨星")
    print("=" * 64)
    all_ok = True
    all_ok &= stage_p0_emoji()
    all_ok &= stage_import_and_wire()
    all_ok &= stage_pytest()
    all_ok &= stage_scorecard()
    all_ok &= stage_runtime_invariants()
    all_ok &= stage_determinism()
    all_ok &= stage_reproducibility_note()

    report_path = os.path.join(REPO, "scripts", "verify_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump({"all_passed": all_ok, "stages": RESULTS}, fh, ensure_ascii=False, indent=2)

    passed = sum(1 for r in RESULTS if r["passed"])
    total = len(RESULTS)
    print("-" * 64)
    print(f"汇总：通过 {passed} / 共 {total}  最终={'PASS' if all_ok else 'FAIL'}")
    print(f"报告已写入：{report_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

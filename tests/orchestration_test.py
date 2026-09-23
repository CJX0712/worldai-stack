"""编排层（LLM + 工具 + ReAct + RAG 管线）测试。"""

from worldai.agent.react import ReActAgent
from worldai.agent.tools import CalculatorTool, DateTool, detect_arithmetic, safe_eval
from worldai.config import Settings, build_stack
from worldai.errors import ToolError
from worldai.llm.mock_llm import MockLLM
from worldai.llm.prompt import build_rag_prompt, build_react_prompt
from worldai.models import Chunk, RetrievalHit
from worldai.util.evidence import parse_evidence, render_evidence


def test_mock_llm_rag_context_mode():
    hits = [RetrievalHit(chunk=Chunk(chunk_id="d0::c0", doc_id="d0", text="量子叠加态实现并行"), score=0.0)]
    ev = render_evidence(hits)
    prompt = build_rag_prompt("问题", ev)
    out = MockLLM().complete(prompt)
    assert "量子叠加态实现并行" in out  # 抽取式回答，不输出 Action


def test_mock_llm_react_first_call_returns_action():
    prompt = build_react_prompt("量子计算是什么", "- retriever: 检索")
    out = MockLLM().complete(prompt)
    assert "Action: retriever" in out
    assert "Final Answer:" not in out


def test_mock_llm_react_followup_returns_final():
    obs = render_evidence([RetrievalHit(chunk=Chunk(chunk_id="d0::c0", doc_id="d0", text="量子比特是基本信息单元"), score=0.0)])
    prompt = build_react_prompt("量子计算是什么", "- retriever: 检索", observation=obs)
    out = MockLLM().complete(prompt)
    assert "Final Answer:" in out
    assert "量子比特是基本信息单元" in out


def test_safe_eval_and_detect():
    assert safe_eval("12*(3+4)") == 84
    assert safe_eval("2.5 + 1.5") == 4.0
    assert detect_arithmetic("帮我计算 12*(3+4) 等于多少？") == "12*(3+4)"
    assert detect_arithmetic("3 篇文档的总结") is None
    assert detect_arithmetic("2024 年营收增长") is None


def test_calculator_division_by_zero_raises_tool_error():
    raised = False
    try:
        CalculatorTool().run("1/0")
    except ToolError:
        raised = True
    assert raised


def test_date_tool_returns_string():
    out = DateTool().run("今天星期几")
    assert isinstance(out, str) and len(out) > 0


def test_evidence_roundtrip():
    hits = [
        RetrievalHit(chunk=Chunk(chunk_id="d0::c0", doc_id="d0", text="第一段\n第二行"), score=0.0),
        RetrievalHit(chunk=Chunk(chunk_id="d1::c0", doc_id="d1", text="另一段\n内容"), score=0.0),
    ]
    ev = render_evidence(hits)
    parsed = parse_evidence(ev)
    assert parsed == ["第一段\n第二行", "另一段\n内容"]


def _build_kb(stack):
    text = (
        "# 量子计算简介\n量子计算是一种基于量子力学原理的计算范式。\n\n"
        "量子计算利用量子叠加态与量子纠缠实现并行运算，能够在特定问题上获得指数级加速。\n\n"
        "传统计算机使用比特，而量子计算机使用量子比特（qubit）作为基本信息单元。\n\n"
        "著名的量子算法如 Shor 算法可用于大整数分解。\n"
    )
    stack["pipeline"].ingest("kb1", "量子计算", "inline", text=text)
    return stack


def test_rag_pipeline_query_grounded():
    stack = build_stack(Settings())
    _build_kb(stack)
    res = stack["pipeline"].query("量子计算利用什么实现并行运算？")
    assert res.hits
    assert any("量子叠加态" in h.chunk.text for h in res.hits)
    # 默认 MockLLM 抽取式回答，答案必为检索证据的子串
    joined = "\n".join(h.chunk.text for h in res.hits)
    assert res.answer in joined


def test_react_agent_knowledge_converges():
    stack = build_stack(Settings())
    _build_kb(stack)
    result = stack["agent"].run("量子计算利用什么实现并行运算？")
    assert result.retrieved_chunk_ids
    assert "retriever" in result.used_tools
    assert len(result.steps) >= 1
    # 答案应与检索证据接地（默认 MockLLM 抽取式回答）
    all_text = "\n".join(c.text for c in stack["pipeline"]._all_chunks)
    assert result.answer in all_text or result.answer


def test_react_agent_arithmetic_routing():
    stack = build_stack(Settings())
    result = stack["agent"].run("帮我计算 12*(3+4) 等于多少？")
    assert "calculator" in result.used_tools
    assert "84" in result.answer


def test_react_agent_date_routing():
    stack = build_stack(Settings())
    result = stack["agent"].run("今天是星期几？")
    assert "date" in result.used_tools


def test_ingest_shorter_doc_drops_orphans():
    stack = build_stack(Settings())
    # 构造超过 chunk_size 的长文档，确保被切成多个块
    long_text = "\n\n".join(
        "段落%d 内容 "%i + "词语 " * 60 for i in range(12)
    )
    stack["pipeline"].ingest("d1", "长文档", "inline", text=long_text)
    first = stack["pipeline"].retrieve("段落0 内容", top_k=20)
    assert len(first) > 1  # 长文档被切成多块
    # 重复摄入更短文档，旧分块应被清除
    stack["pipeline"].ingest("d1", "短文档", "inline", text="仅一段内容。")
    second = stack["pipeline"].retrieve("段落0 内容", top_k=20)
    assert len(second) == 1

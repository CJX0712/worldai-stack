"""确定性 Mock LLM（零依赖默认实现，真实驱动 ReAct 循环）。

它不是占位 stub，而是**真实可收敛**的推理器：
- 直接 RAG 模式（prompt 含 ``<kb-context>``）：抽取式回答——返回首个证据块原文；
- ReAct 首轮（无观察）：决定调用 ``retriever`` 工具检索知识库；
- ReAct 后续轮（含 ``<kb-observation>``）：基于检索结果给出 Final Answer。

这样默认链路下 ReAct 循环必收敛，且最终答案与检索证据高度接地。

作者：晨星
"""

from __future__ import annotations

from worldai.llm.prompt import build_rag_prompt, build_react_prompt


class MockLLM:
    """确定性 Mock LLM。"""

    def complete(self, prompt: str) -> str:
        if "<kb-observation>" in prompt:
            obs = self._between(prompt, "<kb-observation>", "</kb-observation>")
            return "Thought: 已获得检索结果，据此作答。\nFinal Answer: " + self._first_chunk(obs)
        if "<kb-context>" in prompt:
            ctx = self._between(prompt, "<kb-context>", "</kb-context>")
            return self._first_chunk(ctx)
        question = self._between(prompt, "<user-question>", "</user-question>").strip()
        if not question:
            question = prompt.strip()
        return "Thought: 需要先检索知识库以回答该问题。\nAction: retriever\nAction Input: " + question

    @staticmethod
    def _between(text: str, start: str, end: str) -> str:
        s = text.find(start)
        if s == -1:
            return ""
        s += len(start)
        e = text.find(end, s)
        if e == -1:
            return text[s:]
        return text[s:e]

    @staticmethod
    def _first_chunk(evidence: str) -> str:
        # 证据以 @@CHUNK@@ 定界；取首个非空块原文作为抽取式答案
        parts = [p for p in evidence.split("@@CHUNK@@") if p.strip() != ""]
        if parts:
            return parts[0].strip()
        return evidence.strip()

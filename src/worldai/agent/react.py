"""ReAct 智能体（零依赖默认实现）。

核心机制（来自多轮实测）：
- **确定性前置路由**：算术表达式正则检出后直接交给计算器，禁止 LLM 重算
  （小模型算术不可靠，12*(3+4) 都可能算错）；日期类问题同理路由至日期工具；
- **真实 ReAct 循环**：Thought -> Action -> Observation 多步推进，最多 ``max_agent_steps`` 步；
- **工具错误兜底**：工具抛 ``ToolError`` 时作为 Observation 回灌，不中断推理；
- **检索可追溯**：``retriever`` 工具命中的 chunk_id 记录到结果，供评估与审计。

作者：晨星
"""

from __future__ import annotations

import re

from worldai.agent.tools import CalculatorTool, DateTool, detect_arithmetic
from worldai.errors import ToolError
from worldai.llm.prompt import build_react_prompt
from worldai.models import AgentResult, AgentStep

_DATE_RE = re.compile(r"今天|日期|现在|当前|时间|几号|星期|周几|年月日|什么时候")


class ReActAgent:
    """ReAct 工具智能体。"""

    def __init__(self, llm: object, tools: list, settings: object) -> None:
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.settings = settings
        self.max_steps = getattr(settings, "max_agent_steps", 6)

    def run(self, task: str) -> AgentResult:
        used_tools: list[str] = []
        retrieved: list[str] = []

        # ---- 确定性前置路由：算术 ----
        expr = detect_arithmetic(task)
        if expr is not None:
            try:
                obs = CalculatorTool().run(expr)
            except ToolError as exc:
                obs = str(exc)
            used_tools.append("calculator")
            return AgentResult(
                answer=obs,
                steps=[AgentStep(thought="检测到算术表达式，确定性前置路由交由计算器", action="calculator", action_input=expr, observation=obs)],
                used_tools=used_tools,
            )

        # ---- 确定性前置路由：日期 ----
        if _DATE_RE.search(task):
            obs = DateTool().run(task)
            used_tools.append("date")
            return AgentResult(
                answer=obs,
                steps=[AgentStep(thought="检测到时间类问题，路由至日期工具", action="date", action_input=task, observation=obs)],
                used_tools=used_tools,
            )

        # ---- 通用 ReAct 循环 ----
        tool_desc = self._tool_descriptions()
        steps: list[AgentStep] = []
        obs: str | None = None
        for _ in range(self.max_steps):
            prompt = build_react_prompt(task, tool_desc, observation=obs)
            out = self.llm.complete(prompt)
            if "Final Answer:" in out:
                return AgentResult(
                    answer=self._parse_final(out),
                    steps=steps,
                    used_tools=used_tools,
                    retrieved_chunk_ids=retrieved,
                )
            action, action_input = self._parse_action(out)
            tool = self.tools.get(action)
            if tool is None:
                observation = f"未知工具: {action}"
            else:
                try:
                    observation = tool.run(action_input)
                except ToolError as exc:
                    observation = f"工具执行失败: {exc}"
                if tool.name not in used_tools:
                    used_tools.append(tool.name)
                if tool.name == "retriever":
                    retrieved.extend(h.chunk.chunk_id for h in tool.last_hits)
            steps.append(
                AgentStep(
                    thought=self._parse_thought(out),
                    action=action,
                    action_input=action_input,
                    observation=observation,
                )
            )
            obs = observation

        # 步数耗尽：用最后一次观察再请求一次结论
        prompt = build_react_prompt(task, tool_desc, observation=obs)
        out = self.llm.complete(prompt)
        return AgentResult(
            answer=self._parse_final(out),
            steps=steps,
            used_tools=used_tools,
            retrieved_chunk_ids=retrieved,
        )

    def _tool_descriptions(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    @staticmethod
    def _parse_action(out: str) -> tuple[str, str]:
        action = None
        action_input = ""
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("Action:") and action is None:
                action = s[len("Action:"):].strip()
            elif s.startswith("Action Input:"):
                action_input = s[len("Action Input:"):].strip()
        return (action or ""), action_input

    @staticmethod
    def _parse_thought(out: str) -> str:
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("Thought:"):
                return s[len("Thought:"):].strip()
        return ""

    @staticmethod
    def _parse_final(out: str) -> str:
        if "Final Answer:" in out:
            ans = out.split("Final Answer:", 1)[1]
            idx = ans.find("\nThought:")
            if idx != -1:
                ans = ans[:idx]
            return ans.strip()
        # 兜底：排除协议行（action:/thought:/observation:），保留其余作为答案
        lines = [
            ln for ln in out.splitlines()
            if not re.match(r"^(Thought|Action|Action Input|Observation):", ln.strip())
        ]
        return "\n".join(lines).strip()

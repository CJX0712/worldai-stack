"""Prompt 模板（零依赖）。

关键约定（来自实测坑）：
- 上下文/问题用**全局唯一**分隔符 ``<kb-context>`` / ``<kb-observation>`` / ``<user-question>``，
  指令文本中**绝不出现**这些字面量，避免与内容块撞名导致抽取错位；
- 直接 RAG 与 ReAct 两种 prompt 区分清晰，便于 Mock/真实 LLM 一致消费。

作者：晨星
"""

from __future__ import annotations


REACT_SYSTEM = (
    "你是一个严谨的研究助手，可以调用工具来回答问题。\n"
    "当需要调用工具时，严格按如下格式输出，不要输出多余内容：\n"
    "Thought: 你的推理\n"
    "Action: 工具名称\n"
    "Action Input: 工具输入\n"
    "当你已经得到足够信息可以回答时，输出：\n"
    "Thought: 你的推理\n"
    "Final Answer: 最终答案\n"
)


def build_react_prompt(question: str, tool_descriptions: str, observation: str | None = None) -> str:
    # 注意：指令文本中绝不出现 <user-question>/<kb-observation> 字面量，
    # 否则会与真实内容块分隔符撞名导致抽取错位（已踩坑）。
    prompt = REACT_SYSTEM + "\n\n可用工具：\n" + tool_descriptions + "\n"
    prompt += (
        "\n下方用专用标记包裹的是用户问题，请针对该问题作答，"
        "勿将标记文本本身当作检索内容：\n"
        "<user-question>\n" + question + "\n</user-question>\n"
    )
    if observation is not None:
        prompt += "\n下方用专用标记包裹的是已获得的检索结果：\n"
        prompt += "<kb-observation>\n" + observation + "\n</kb-observation>\n"
    return prompt


def build_rag_prompt(question: str, evidence: str) -> str:
    # 指令中不出现 <kb-context> 字面量，仅作为真实资料块分隔符
    prompt = (
        "你是知识库问答助手。请仅依据下方用专用标记包裹的资料块回答用户问题；"
        "若资料不足以回答，请明确说明无法回答。\n\n"
        "<kb-context>\n" + evidence + "\n</kb-context>\n\n"
        "用户问题：\n" + question + "\n\n回答："
    )
    return prompt

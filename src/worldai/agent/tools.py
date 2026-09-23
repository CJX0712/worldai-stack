"""智能体工具集（零依赖）。

包含：
- :class:`CalculatorTool`：安全算术计算（AST 白名单求值，除零/溢出归一为 ToolError）；
- :class:`DateTool`：返回当前日期时间；
- :class:`RetrieverTool`：调用 RAG 管线检索，返回 ``@@CHUNK@@`` 定界证据。

确定性前置路由所需的算术检测（:func:`detect_arithmetic`）与求值（:func:`safe_eval`）
也在此实现，供智能体在调用 LLM 前先处理数值类问题，避免小模型重算出错。

作者：晨星
"""

from __future__ import annotations

import ast
import datetime
import re

from worldai.errors import ToolError
from worldai.util.evidence import render_evidence

# 全角运算符/括号/中文运算符归一为半角，避免中文问句里的符号阻断路由
_WS_MAP = {
    "＋": "+", "－": "-", "×": "*", "÷": "/",
    "（": "(", "）": ")",
    "加": "+", "减": "-", "乘": "*", "除": "/",
}
# 疑问/指令类助词，非运算符，路由前整体剥离
_PARTICLE = "等于多少计算请帮问我等的结果呀呢啊吧什么为何如何怎么"


def _normalize_expr(s: str) -> str:
    for k, v in _WS_MAP.items():
        s = s.replace(k, v)
    s = s.strip()
    # 剥离句末全角/半角标点（绝不能包含运算符字符，否则会切坏合法表达式）
    s = re.sub(r"[？！。．，、\s]+$", "", s)
    return s


def _is_pure_arithmetic(s: str) -> bool:
    if not s:
        return False
    if not re.fullmatch(r"[0-9+\-*/().]+", s):
        return False
    if not any(c in "+-*/" for c in s):
        return False
    if sum(1 for c in s if c.isdigit()) == 0:
        return False
    if s.count("(") != s.count(")"):
        return False
    return True


def detect_arithmetic(text: str) -> str | None:
    """若文本整体可归约为可计算算术表达式，返回归一化后的表达式；否则返回 None。"""
    s = _normalize_expr(text)
    cleaned = "".join(ch for ch in s if ch not in _PARTICLE and not ch.isspace())
    if _is_pure_arithmetic(cleaned):
        return cleaned
    return None


def safe_eval(expr: str) -> float:
    """安全求值算术表达式（仅允许数字与 + - * / 及括号）。"""
    try:
        node = ast.parse(_normalize_expr(expr), mode="eval")
        return _eval_node(node.body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        raise ToolError(f"无法计算的算术表达式: {expr!r} ({exc})") from exc


def _eval_node(node: ast.AST):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("仅支持数值常量")
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +_eval_node(node.operand)
        raise ValueError("不支持的一元运算")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError("除零")
            return left / right
        raise ValueError("不支持的二元运算")
    raise ValueError("不支持的表达式结构")


class CalculatorTool:
    """安全算术计算器。"""

    name = "calculator"
    description = "计算算术表达式，例如 12*(3+4)。输入为表达式字符串。"

    def run(self, action_input: str) -> str:
        value = safe_eval(action_input)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return f"计算结果: {value}"


class DateTool:
    """返回当前日期时间。"""

    name = "date"
    description = "返回当前系统日期与时间（用于回答与时间相关的问题）。"

    def run(self, action_input: str) -> str:
        now = datetime.datetime.now()
        return now.strftime("当前时间: %Y-%m-%d %H:%M:%S %A")


class RetrieverTool:
    """知识库检索工具：调用 RAG 管线并返回定界证据。"""

    name = "retriever"
    description = "在知识库中检索与问题相关的文档片段。输入为用户问题。"

    def __init__(self, pipeline: object, top_k: int = 6) -> None:
        self.pipeline = pipeline
        self.top_k = top_k
        self.last_hits: list = []

    def run(self, action_input: str) -> str:
        hits = self.pipeline.retrieve(action_input, self.top_k)
        self.last_hits = hits
        return render_evidence(hits)

"""证据渲染与解析（零依赖）。

- 检索命中统一渲染为 ``@@CHUNK@@`` 定界文本，避免换行导致「块仅首行进入模型」
  的上下文丢失；
- 解析为分块文本列表时做**往返保真**断言（render -> parse 还原一致）。

作者：晨星
"""

from __future__ import annotations

from worldai.models import RetrievalHit

_CHUNK_SEP = "@@CHUNK@@"


def render_evidence(hits: list[RetrievalHit]) -> str:
    """将命中列表渲染为定界文本。"""
    return "".join(_CHUNK_SEP + h.chunk.text for h in hits)


def parse_evidence(text: str) -> list[str]:
    """从渲染文本解析出分块文本列表（去空前导段）。"""
    return [seg for seg in text.split(_CHUNK_SEP) if seg != ""]

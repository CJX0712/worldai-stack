"""语义分块器（带标题继承）。

设计要点（来自多轮实测踩坑）：
- **标题继承**：每个块携带「起始位置之前最近的标题」；若无则继承文档标题。
  用「最后一个 start <= position 的标题」做语义，避免首个块丢失标题。
- **重叠窗口**：块尾保留 ``overlap`` 字符进入下一块，提升跨块语义连续性。
- **确定性**：纯偏移扫描，与输入长度无关地可复现。

作者：晨星
"""

from __future__ import annotations

import re

from worldai.models import Chunk

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")


class SemanticChunker:
    """按字符窗口切分，并在每个块上标注其所属标题。"""

    def __init__(self, chunk_size: int = 480, overlap: int = 80) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap 必须 >=0 且 < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, title: str, doc_id: str) -> list[Chunk]:
        lines = text.split("\n")

        # 预扫描所有标题行的 (offset, heading_text)
        heading_marks: list[tuple[int, str]] = []
        offset = 0
        line_offsets: list[int] = []
        for line in lines:
            line_offsets.append(offset)
            m = _HEADING_RE.match(line)
            if m:
                heading_marks.append((offset, m.group(2).strip()))
            offset += len(line) + 1  # 还原被 split 吃掉的行尾换行

        def heading_for(pos: int) -> str:
            # 最后一个 start <= pos 的标题；无则文档标题
            h = title
            for off, ht in heading_marks:
                if off <= pos:
                    h = ht
                else:
                    break
            return h

        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_len = 0
        start_offset = 0

        def flush() -> None:
            nonlocal buf, buf_len, start_offset
            if not buf:
                return
            chunk_text = "\n".join(buf)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::c{len(chunks)}",
                    doc_id=doc_id,
                    text=chunk_text,
                    heading=heading_for(start_offset),
                    position=start_offset,
                )
            )
            tail = chunk_text[-self.overlap:] if self.overlap else ""
            if tail:
                tail_start = start_offset + len(chunk_text) - len(tail)
                buf = [tail]
                buf_len = len(tail)
                start_offset = tail_start
            else:
                buf = []
                buf_len = 0

        for off, line in zip(line_offsets, lines):
            line_len = len(line)
            if buf and buf_len + line_len + 1 > self.chunk_size:
                flush()
            if not buf:
                start_offset = off
            buf.append(line)
            buf_len += line_len + (1 if len(buf) > 1 else 0)

        flush()
        if not chunks:
            # 空/纯空白文档：保底返回一个整块，避免下游索引为空
            chunks.append(
                Chunk(chunk_id=f"{doc_id}::c0", doc_id=doc_id, text=text, heading=title, position=0)
            )
        return chunks

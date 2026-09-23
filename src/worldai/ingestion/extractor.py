"""文本抽取器。

- :class:`TextExtractor`：统一入口，按文件扩展名分流；非文件路径则当作原始文本原样返回
  （零依赖，纯标准库）。
- :class:`PdfExtractor`：基于 ``pypdf`` 抽取 PDF 正文（惰性导入，缺失依赖时清晰报错）。

作者：晨星
"""

from __future__ import annotations

import os

from worldai.errors import IngestError


class TextExtractor:
    """纯文本 / Markdown 抽取，并分流 PDF。"""

    def extract(self, source: str) -> str:
        """抽取文本。

        ``source`` 为存在的文件路径时按扩展名处理；否则视为原始文本直接返回。
        """
        if os.path.isfile(source):
            ext = os.path.splitext(source)[1].lower()
            if ext == ".pdf":
                return PdfExtractor().extract(source)
            try:
                with open(source, "r", encoding="utf-8") as handle:
                    return handle.read()
            except UnicodeDecodeError:
                # 兜底：以 latin-1 读取，避免二进制/异编码文件直接崩溃
                with open(source, "r", encoding="latin-1") as handle:
                    return handle.read()
        return source


class PdfExtractor:
    """PDF 正文抽取（依赖 ``pypdf``）。"""

    def extract(self, path: str) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise IngestError(
                "PDF 抽取需要 pypdf，请先 `pip install pypdf` 或改用默认文本摄入。"
            ) from exc
        if not os.path.isfile(path):
            raise IngestError(f"PDF 文件不存在：{path}")
        try:
            reader = PdfReader(path)
            pages = [(p.extract_text() or "") for p in reader.pages]
            return "\n".join(pages).strip()
        except Exception as exc:  # pragma: no cover - 取决于具体文件
            raise IngestError(f"PDF 抽取失败：{exc}") from exc

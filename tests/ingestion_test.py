"""摄入层测试：文本抽取 + 语义分块（标题继承 / 重叠 / 内容保真）。"""

import os
import tempfile

from worldai.ingestion.chunker import SemanticChunker
from worldai.ingestion.extractor import PdfExtractor, TextExtractor


def test_text_extractor_raw_text_passthrough():
    assert TextExtractor().extract("hello world") == "hello world"


def test_text_extractor_reads_txt_file():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("# 标题\n一些正文内容。")
        path = fh.name
    try:
        out = TextExtractor().extract(path)
        assert "标题" in out and "一些正文内容" in out
    finally:
        os.unlink(path)


def test_pdf_extractor_missing_file_raises():
    raised = False
    try:
        PdfExtractor().extract("/nonexistent/path/file.pdf")
    except Exception as exc:  # noqa: BLE001
        raised = True
        assert "PDF" in str(exc)
    assert raised


def test_chunker_heading_inheritance():
    text = (
        "# 引言\n一段简短的引言。\n\n"
        "# 方法\n" + "词 " * 400 + "\n\n"
        "# 结果\n" + "结果 " * 400
    )
    chunks = SemanticChunker(chunk_size=200, overlap=40).chunk(text, "文档", "d1")
    assert len(chunks) > 1
    # 首个块起始于文档开头，应继承首个标题
    assert chunks[0].heading == "引言"
    # 起始位置落在「方法」节之后的块应继承「方法」
    methods_pos = text.index("# 方法")
    methods_chunk = next(c for c in chunks if c.position >= methods_pos)
    assert methods_chunk.heading == "方法"


def test_chunker_empty_document_has_one_chunk():
    chunks = SemanticChunker().chunk("", "T", "d")
    assert len(chunks) == 1
    assert chunks[0].heading == "T"
    assert chunks[0].text == ""


def test_chunker_nonuniform_preserves_all_paragraphs():
    # 刻意非均匀段落长度，避免周期性分块假象
    paras = ["段落%d " % i + "字 " * (30 + (i % 5) * 25) for i in range(20)]
    text = "\n\n".join(paras)
    chunks = SemanticChunker(chunk_size=300, overlap=50).chunk(text, "T", "d")
    joined = "\n".join(c.text for c in chunks)
    for p in paras:
        assert p in joined


def test_chunker_overlap_keeps_continuity():
    text = "甲 " * 300 + "\n" + "乙 " * 300
    chunks = SemanticChunker(chunk_size=200, overlap=60).chunk(text, "T", "d")
    # 重叠区应使相邻块存在公共子串
    assert len(chunks) >= 2
    tail = chunks[0].text[-40:]
    assert tail in chunks[1].text

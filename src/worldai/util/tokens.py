"""统一分词：拉丁词 + 中文二元文法（bigram）。

设计要点：
- 拉丁字母数字串作为整词（小写）；
- 中日韩统一表意文字取**字符二元文法**，使中文短文本的嵌入/检索具备合理的
  词法重叠信号（中文无天然空格，必须靠 bigram 获得局部共现）；
- 纯函数、确定性，零依赖。

作者：晨星
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """将文本切分为 token 列表。"""
    if not text:
        return []
    low = text.lower()
    toks: list[str] = _LATIN.findall(low)
    cjk = "".join(_CJK.findall(low))
    n = len(cjk)
    if n >= 2:
        for i in range(n - 1):
            toks.append("cjk:" + cjk[i : i + 2])
    elif n == 1:
        toks.append("cjk:" + cjk)
    return toks


def term_frequencies(text: str) -> dict[str, int]:
    """返回 token -> 词频。"""
    tf: dict[str, int] = {}
    for tok in tokenize(text):
        tf[tok] = tf.get(tok, 0) + 1
    return tf

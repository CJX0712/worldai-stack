"""P0 字符门禁：全仓扫描源码中的 emoji / 符号字面量。

动机（实测坑）：源码里写了 BMP 符号字面量（如 ☀-➿），经 GBK 代码页落盘后会被
mangle，导致正则/字符串比对诡异失败。因此**源文件里绝不放 emoji/符号字面量**，本脚本
用纯 ``ord()`` 码点范围检测，不放过任何符号。仅扫描 ``.py`` 文件。

作者：晨星
"""

from __future__ import annotations

import os
import sys

# 禁止的码点区间（符号 / 箭头 / 几何图形 / 装饰符 / emoji / 变体选择器）
_FORBIDDEN_RANGES = [
    (0x2190, 0x21FF),  # 箭头
    (0x2300, 0x23FF),  # 技术符号
    (0x2460, 0x24FF),  # 带圈字母数字
    (0x2500, 0x25FF),  # 制表符 / 几何图形
    (0x2600, 0x27BF),  # 杂项符号 + 丁巴特符号
    (0x2B00, 0x2BFF),  # 杂项符号与箭头
    (0x1F000, 0x1FAFF),  # emoji
    (0xFE0F, 0xFE0F),  # 变体选择器
]


def _is_forbidden(cp: int) -> bool:
    return any(start <= cp <= end for start, end in _FORBIDDEN_RANGES)


def scan_file(path: str) -> list[dict]:
    findings: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            for col, ch in enumerate(line, 1):
                cp = ord(ch)
                if _is_forbidden(cp):
                    findings.append(
                        {"file": path, "line": lineno, "col": col, "char": ch, "codepoint": hex(cp)}
                    )
    return findings


def scan_directory(root: str) -> list[dict]:
    all_findings: list[dict] = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                all_findings.extend(scan_file(os.path.join(dirpath, name)))
    return all_findings


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    targets = [os.path.join(repo_root, "src"), os.path.join(repo_root, "tests")]
    findings: list[dict] = []
    for t in targets:
        if os.path.isdir(t):
            findings.extend(scan_directory(t))
    if findings:
        print("[P0] 发现源码中的 emoji / 符号字面量（禁止）：")
        for f in findings:
            print(f"  {f['file']}:{f['line']}:{f['col']}  {f['char']!r} ({f['codepoint']})")
        return 1
    print(f"[P0] 字符门禁通过：扫描 {len(findings)} 处违规（0 findings）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""WorldAI 服务启动入口（开发/演示用）。

用法：
    python scripts/serve.py                 # 默认零依赖配置，监听 :8000
    WORLDAI_EMBEDDER=fastembed python scripts/serve.py --port 8080

作者：晨星
"""

from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from worldai.api.app import create_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="WorldAI REST 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        print("需要 uvicorn：pip install uvicorn")
        return 1
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

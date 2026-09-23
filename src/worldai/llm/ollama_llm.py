"""Ollama LLM（生产实现，HTTP 调用本地/远端 Ollama）。

- 复用 Ollama 生态（兼容 OpenAI 风格的大量本地模型）；
- **关键坑**：httpx 默认 ``trust_env=True`` 会把本机流量也送进系统 SOCKS/HTTP 代理，
  导致 ``localhost`` 连接被重置（WinError 10054）；所有指向本机的请求一律
  ``trust_env=False``；
- 构造不触网，仅在 ``complete`` 时发起请求。

作者：晨星
"""

from __future__ import annotations

from worldai.errors import ProviderError


class OllamaLLM:
    """Ollama HTTP LLM（生产实现）。"""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:0.5b", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("OllamaLLM 需要 httpx：pip install httpx") from exc
        # 指向本机一律绕过系统代理，避免 localhost 连接被重置
        with httpx.Client(timeout=self.timeout, trust_env=False) as client:
            resp = client.post(
                self.base_url + "/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

"""llama.cpp LLM（生产实现，本地 GGUF 推理）。

- 复用 ``llama-cpp-python``，在 CPU/GPU 上本地运行 GGUF 量化模型，零外部服务依赖；
- 模型**惰性加载**，构造仅记录路径，首次 ``complete`` 时载入权重；
- 未配置模型路径时由 :meth:`_ensure` 抛出 :class:`ProviderError`。

作者：晨星
"""

from __future__ import annotations

from worldai.errors import ProviderError


class LlamaCppLLM:
    """llama.cpp 本地 LLM（生产实现）。"""

    def __init__(self, model_path: str = "", n_ctx: int = 2048, n_threads: int | None = None) -> None:
        if not model_path:
            # 允许先构造、后通过属性补路径；但真正推理前必须就绪
            pass
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None

    def _ensure(self) -> None:
        if self._llm is None:
            if not self.model_path:
                raise ProviderError("LlamaCppLLM 未配置模型路径：设置 WORLDAI_LLAMACPP_MODEL_PATH")
            try:
                from llama_cpp import Llama
            except ImportError as exc:  # pragma: no cover
                raise ProviderError("LlamaCppLLM 需要 llama-cpp-python：pip install llama-cpp-python") from exc
            kwargs = {"model_path": self.model_path, "n_ctx": self.n_ctx}
            if self.n_threads is not None:
                kwargs["n_threads"] = self.n_threads
            self._llm = Llama(**kwargs)

    def complete(self, prompt: str) -> str:
        self._ensure()
        out = self._llm(prompt, max_tokens=256)  # type: ignore[union-attr]
        return out["choices"][0]["text"]

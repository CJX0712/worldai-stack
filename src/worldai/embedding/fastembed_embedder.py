"""FastEmbed 嵌入器（生产实现，基于 ONNX 文本嵌入模型）。

- 复用业界领先的 ``fastembed``（Qdrant 出品，ONNX 运行时，无需 PyTorch）；
- 模型**惰性加载**：构造不下载，首次 ``embed`` 时才拉取并缓存权重；
- 默认 ``BAAI/bge-small-zh-v1.5``，中英双语，适合中文知识库；
- 缺失依赖或离线时由 :meth:`_ensure` 抛出 :class:`ProviderError`，清晰可诊断。

作者：晨星
"""

from __future__ import annotations

from worldai.errors import ProviderError


class FastEmbedEmbedder:
    """FastEmbed 嵌入器（生产实现）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._dim: int | None = None

    def _ensure(self) -> None:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover
                raise ProviderError(
                    "FastEmbedEmbedder 需要 fastembed：pip install fastembed"
                ) from exc
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
            probe = next(iter(self._model.embed(["维度探测"])))
            self._dim = int(probe.shape[0])

    def dim(self) -> int:
        if self._dim is None:
            self._ensure()
        return self._dim  # type: ignore[return-value]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        out = []
        for vec in self._model.embed(texts):  # type: ignore[union-attr]
            out.append([float(x) for x in vec])
        return out

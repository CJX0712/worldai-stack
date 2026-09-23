"""内存文档仓储（零依赖默认实现）。

- 以 ``doc_id`` 为主键保存 :class:`Document`（含其分块）；
- 进程内存储，确定性、零依赖，支撑默认链路与单测；
- 生产环境可替换为 Postgres / 对象存储实现（同接口）。

作者：晨星
"""

from __future__ import annotations

from worldai.models import Document


class MemoryRepository:
    """内存文档仓储。"""

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}

    def put(self, doc: Document) -> None:
        self._docs[doc.doc_id] = doc

    def get(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def all(self) -> list[Document]:
        return list(self._docs.values())

    def delete(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)

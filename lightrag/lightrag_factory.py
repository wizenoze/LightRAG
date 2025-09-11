from __future__ import annotations

import traceback
import asyncio
import configparser
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    cast,
    final,
    Literal,
    Optional,
    List,
    Dict,
)
from lightrag.constants import (
    DEFAULT_MAX_GLEANING,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_MIN_RERANK_SCORE,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_SUMMARY_CONTEXT_SIZE,
    DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    DEFAULT_MAX_ASYNC,
    DEFAULT_MAX_PARALLEL_INSERT,
    DEFAULT_MAX_GRAPH_NODES,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_EMBEDDING_TIMEOUT,
)
from lightrag.utils import get_env_value

from lightrag.kg import (
    STORAGES,
    verify_storage_implementation,
)


from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
    get_graph_db_lock,
    get_data_init_lock,
)
from . import LightRAG

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
    DeletionResult,
    OllamaServerInfos,
)
from .namespace import NameSpace
from .operate import (
    chunking_by_token_size,
    extract_entities,
    merge_nodes_and_edges,
    kg_query,
    naive_query,
    _rebuild_knowledge_from_chunks,
)
from .constants import GRAPH_FIELD_SEP
from .utils import (
    Tokenizer,
    TiktokenTokenizer,
    EmbeddingFunc,
    always_get_an_event_loop,
    compute_mdhash_id,
    lazy_external_import,
    priority_limit_async_func_call,
    get_content_summary,
    sanitize_text_for_encoding,
    check_storage_env_vars,
    generate_track_id,
    logger,
)
from .types import KnowledgeGraph
from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# TODO: TO REMOVE @Yannick
config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

_DEFAULT_WORKSPACE_NAME : str  = "_DEFAULT_"

@final
@dataclass
class LightRAGFactory:
    lightrag_args: field(default_factory=dict)

    working_dir: str = field(default="./rag_storage")
    """Directory where cache and temporary files are stored."""

    ollama_server_infos: Optional[OllamaServerInfos] = field(default=None)
    """Configuration for Ollama server information."""


    def __post_init__(self):
        from lightrag.kg.shared_storage import (
            initialize_share_data,
        )
        self._instances: dict[str, LightRAG] = {}

        # Initialize ollama_server_infos if not provided
        if self.ollama_server_infos is None:
            self.ollama_server_infos = OllamaServerInfos()
        initialize_share_data()
        self.create(workspace=_DEFAULT_WORKSPACE_NAME, reuse=True)

    def _build_kwargs(self, workspace: str) -> dict[str, Any]:
        args = dict(self.lightrag_args)
        # Ensure workspace and working_dir are set consistently
        args["workspace"] = workspace
        args["working_dir"] = self._compute_working_dir(workspace, args)
        return args

    def _compute_working_dir(self, workspace: str, effective_kwargs: dict[str, Any]) -> str:
        # Prefer factory working_dir_base, then kwargs. Default to "./rag_storage".
        base_dir = self.working_dir or effective_kwargs.get("working_dir") or "./rag_storage"
        if workspace:
            return os.path.join(base_dir, workspace)
        return base_dir

    def create(
            self,
            workspace: str,
            reuse: bool = False,
    ) -> LightRAG:
        """
        Create a LightRAG instance for the given workspace.

        - override_kwargs: per-instance overrides merged on top of base_kwargs.
        - reuse: if True, cache and return the same instance for subsequent calls.
        """
        if reuse and workspace in self._instances:
            return self._instances[workspace]
        kwargs = self._build_kwargs(workspace)
        instance = LightRAG(**kwargs)
        async def asyncinit_storages():
            await instance.initialize_storages()
        asyncio.run(asyncinit_storages())
        if reuse:
            self._instances[workspace] = instance

        return instance

    def get(self, workspace: str) -> LightRAG:
        """
        Get a cached LightRAG instance for the workspace, creating it if needed.
        """
        if workspace in self._instances:
            return self._instances[workspace]
        return self.create(workspace, reuse=True)

    async def getDefault(self) -> LightRAG:
        self.get(_DEFAULT_WORKSPACE_NAME)


    async def initialize_storages(self):
        for instance in self._instances.values():
            await instance.initialize_storages()

    async def check_and_migrate_data(self):
        for instance in self._instances.values():
            await instance.check_and_migrate_data()

    async def finalize_storages(self):
        for instance in self._instances.values():
            instance.finalize_storages()

    def effective_workspace(workspace : str) -> str :
        workspace




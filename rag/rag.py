import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Type, cast

from .llm import hf_embedding, hf_model_complete
from .operate import (
    chunking_by_token_size,
    kg_query, naive_query, llm_decomposition, extract_entities, extract_entities_only, texts_summary,
    _merge_nodes_then_upsert, _merge_edges_then_upsert,
)

from .utils import (
    EmbeddingFunc,
    compute_mdhash_id,
    limit_async_func_call,
    convert_response_to_json,
    add_period_if_needed,
    logger,
    set_logger,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    StorageNameSpace,
    QueryParam,
)

from .storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)


def lazy_external_import(module_name: str, class_name: str):
    """Lazily import a class from an external module based on the package of the caller."""

    # Get the caller's module and package
    import inspect

    caller_frame = inspect.currentframe().f_back
    module = inspect.getmodule(caller_frame)
    package = module.__package__ if module else None

    def import_class(*args, **kwargs):
        import importlib

        # Import the module using importlib
        module = importlib.import_module(module_name, package=package)

        # Get the class from the module and instantiate it
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)

    return import_class


Neo4JStorage = lazy_external_import(".kg.neo4j_impl", "Neo4JStorage")
OracleKVStorage = lazy_external_import(".kg.oracle_impl", "OracleKVStorage")
OracleGraphStorage = lazy_external_import(".kg.oracle_impl", "OracleGraphStorage")
OracleVectorDBStorage = lazy_external_import(".kg.oracle_impl", "OracleVectorDBStorage")
MilvusVectorDBStorge = lazy_external_import(".kg.milvus_impl", "MilvusVectorDBStorge")
MongoKVStorage = lazy_external_import(".kg.mongo_impl", "MongoKVStorage")
ChromaVectorDBStorage = lazy_external_import(".kg.chroma_impl", "ChromaVectorDBStorage")
TiDBKVStorage = lazy_external_import(".kg.tidb_impl", "TiDBKVStorage")
TiDBVectorDBStorage = lazy_external_import(".kg.tidb_impl", "TiDBVectorDBStorage")
AGEStorage = lazy_external_import(".kg.age_impl", "AGEStorage")


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


@dataclass
class KGSRAG:
    working_dir: str = field(
        default_factory=lambda: f"./rag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    dataset: str = None
    # Default not to use embedding cache
    embedding_cache_config: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    kv_storage: str = field(default="JsonKVStorage")
    vector_storage: str = field(default="NanoVectorDBStorage")
    graph_storage: str = field(default="NetworkXStorage")

    current_log_level = logger.level
    log_level: str = field(default=current_log_level)

    # text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    embedding_func: EmbeddingFunc = field(default_factory=lambda: hf_embedding)
    # embedding_func: EmbeddingFunc = field(default_factory=lambda: openai_embedding)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM
    llm_model_func: callable = hf_model_complete  # hf_model_complete#
    llm_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # 'meta-llama/Llama-3.2-1B'#'google/gemma-2-2b-it'
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16
    llm_model_kwargs: dict = field(default_factory=dict)

    # storage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)

    enable_llm_cache: bool = True

    # extension
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = convert_response_to_json

    def __post_init__(self):
        log_file = os.path.join("rag.log")
        set_logger(log_file)
        logger.setLevel(self.log_level)

        logger.info(f"Logger initialized for working directory: {self.working_dir}")

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self).items()])
        logger.debug(f"KGSRAG init with param:\n  {_print_config}\n")

        # @TODO: should move all storage setup here to leverage initial start params attached to self.

        self.key_string_value_json_storage_cls: Type[BaseKVStorage] = (self._get_storage_class()[self.kv_storage])
        self.vector_db_storage_cls: Type[BaseVectorStorage] = self._get_storage_class()[self.vector_storage]
        self.graph_storage_cls: Type[BaseGraphStorage] = self._get_storage_class()[self.graph_storage]

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache",
                global_config=asdict(self),
                embedding_func=None,
            )
            if self.enable_llm_cache
            else None
        )
        self.embedding_func = limit_async_func_call(self.embedding_func_max_async)(self.embedding_func)

        ####
        # add embedding func by walter
        ####
        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunks_kv = self.key_string_value_json_storage_cls(
            namespace="text_chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        ####
        # add embedding func by walter over
        ####

        self.entities_kv = self.key_string_value_json_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.relations_kv = self.key_string_value_json_storage_cls(
            namespace="relations",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunks_kv = self.key_string_value_json_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunks_summary_kv = self.key_string_value_json_storage_cls(
            namespace="chunks_summary",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.statements_kv = self.key_string_value_json_storage_cls(
            namespace="statements",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.questions_kv = self.key_string_value_json_storage_cls(
            namespace="questions",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )

        self.chunks_vdb = self.vector_db_storage_cls(
            namespace="chunks",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.chunks_summary_vdb = self.vector_db_storage_cls(
            namespace="chunks_summary",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
        )
        self.entities_vdb = self.vector_db_storage_cls(
            namespace="entities",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"entity_name", "description"},
        )
        self.relations_vdb = self.vector_db_storage_cls(
            namespace="relations",
            global_config=asdict(self),
            embedding_func=self.embedding_func,
            meta_fields={"src_id", "tgt_id", "description"},
        )

        # todo _insert_done()

        self.llm_model_func = limit_async_func_call(self.llm_model_max_async)(
            partial(
                self.llm_model_func,
                hashing_kv=self.llm_response_cache,
                **self.llm_model_kwargs,
            )
        )

    def _get_storage_class(self) -> Type[BaseGraphStorage]:
        return {
            # kv storage
            "JsonKVStorage": JsonKVStorage,
            "OracleKVStorage": OracleKVStorage,
            "MongoKVStorage": MongoKVStorage,
            "TiDBKVStorage": TiDBKVStorage,
            # vector storage
            "NanoVectorDBStorage": NanoVectorDBStorage,
            "OracleVectorDBStorage": OracleVectorDBStorage,
            "MilvusVectorDBStorge": MilvusVectorDBStorge,
            "ChromaVectorDBStorage": ChromaVectorDBStorage,
            "TiDBVectorDBStorage": TiDBVectorDBStorage,
            # graph storage
            "NetworkXStorage": NetworkXStorage,
            "Neo4JStorage": Neo4JStorage,
            "OracleGraphStorage": OracleGraphStorage,
            "AGEStorage": AGEStorage,
            # "ArangoDBStorage": ArangoDBStorage
        }

    def insert(self, string_or_strings):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.ainsert(string_or_strings))

    async def ainsert(self, string_or_strings):
        update_storage = False
        try:
            if isinstance(string_or_strings, str):
                string_or_strings = [string_or_strings]

            new_docs = {
                compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                for c in string_or_strings
            }
            _add_doc_keys = await self.full_docs.filter_keys(list(new_docs.keys()))
            new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
            if not len(new_docs):
                logger.warning("All docs are already in the storage")
                return
            update_storage = True
            logger.info(f"[New Docs] inserting {len(new_docs)} docs")

            inserting_chunks = {}
            for doc_key, doc in tqdm_async(new_docs.items(), desc="Chunking documents", unit="doc"):
                chunks = {
                    compute_mdhash_id(dp["content"], prefix="chunk-"): {
                        **dp,
                        "full_doc_id": doc_key,
                    }
                    for dp in chunking_by_token_size(
                        doc["content"],
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tiktoken_model=self.tiktoken_model_name,
                    )
                }
                inserting_chunks.update(chunks)
            _add_chunk_keys = await self.chunks_kv.filter_keys(list(inserting_chunks.keys()))
            inserting_chunks = {k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys}
            if not len(inserting_chunks):
                logger.warning("All chunks are already in the storage")
                return
            logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
            logger.info("[Information Extraction]...")
            await extract_entities_only(
                inserting_chunks,
                chunks_kv=self.chunks_kv,
                entities_kv=self.entities_kv,
                relations_kv=self.relations_kv,
                global_config=asdict(self),
            )
            # logger.info("[Text Summary]...")
            # await texts_summary(
            #     inserting_chunks,
            #     chunks_summary_kv=self.chunks_summary_kv,
            #     global_config=asdict(self),
            # )
            await self.full_docs.upsert(new_docs)
            await self.chunks_kv.upsert(inserting_chunks)
        finally:
            if update_storage:
                await self._insert_done()

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.llm_response_cache,
            self.entities_kv,
            self.relations_kv,
            self.chunks_kv,
            self.chunks_summary_kv,
            self.chunk_entity_relation_graph,
            self.chunks_vdb,
            self.chunks_summary_vdb,
            self.entities_vdb,
            self.relations_vdb,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def query(self, query: str, param: QueryParam = QueryParam()):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aquery(query, param))

    async def aquery(self, query: str, param: QueryParam = QueryParam(), query_type=None):
        query_list, high_list, lower_list, answer_list = await self.query_decomposition(query, param)

        if param.mode in ["perfect", "keyword", "statement", "all"]:
            response = await kg_query(
                query, query_list, high_list, lower_list, answer_list,
                self.chunk_entity_relation_graph, self.chunks_vdb, self.chunks_summary_vdb,
                self.entities_vdb, self.relations_vdb, self.chunks_kv, self.entities_kv, self.relations_kv,
                param, asdict(self), query_type=query_type, hashing_kv=self.llm_response_cache, cal_tokens=False
            )
        elif param.mode == "naive":
            response = await naive_query(
                query, self.chunks_vdb, self.chunks_kv,
                param, asdict(self), query_type=query_type, hashing_kv=self.llm_response_cache, cal_tokens=False
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        await self._query_done()
        return response

    async def _query_done(self):
        tasks = []
        for storage_inst in [self.llm_response_cache]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    async def query_decomposition(self, query: str, param: QueryParam = QueryParam()):
        if param.decomposition in "statement":
            query_list, high_list, low_list, answer_list \
                = await llm_decomposition(query, param, self.statements_kv, asdict(self))
        elif param.decomposition == "question":
            query_list, high_list, low_list, answer_list \
                = await llm_decomposition(query, param, self.questions_kv, asdict(self))
        else:
            raise ValueError(f"Unknown mode {param.decomposition}")
        return query_list, high_list, low_list, answer_list

    def delete_by_entity(self, entity_name: str):
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_entity(self, entity_name: str):
        entity_name = f'"{entity_name.upper()}"'

        try:
            await self.entities_kv.delete_entity(entity_name)
            await self.relations_kv.delete_relation(entity_name)
            await self.chunk_entity_relation_graph.delete_node(entity_name)

            logger.info(
                f"Entity '{entity_name}' and its relations have been deleted."
            )
            await self._delete_by_entity_done()
        except Exception as e:
            logger.error(f"Error while deleting entity '{entity_name}': {e}")

    async def _delete_by_entity_done(self):
        tasks = []
        for storage_inst in [
            self.entities_kv,
            self.relations_kv,
            self.chunk_entity_relation_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)

    def generate_db(self):
        loop = always_get_an_event_loop()
        loop.run_until_complete(self.agenerate_db())

    async def agenerate_db(self):
        entities_kv = self.entities_kv
        relations_kv = self.relations_kv
        chunks_kv = self.chunks_kv
        chunks_summary_kv = self.chunks_summary_kv
        entities_kv_all = await entities_kv.all_data()
        relations_kv_all = await relations_kv.all_data()
        chunks_kv_all = await chunks_kv.all_data()
        chunks_summary_kv_all = await chunks_summary_kv.all_data()

        data_for_vdb = {
            compute_mdhash_id(chunk_id + dp["entity_name"], prefix="ent-"): {
                "entity_name": dp["entity_name"],
                "description": dp["description"],
                "content": dp["entity_name"] + " " + add_period_if_needed(dp["description"]),
            } for chunk_id in entities_kv_all for dp in entities_kv_all[chunk_id][0]
        }
        await self.entities_vdb.upsert(data_for_vdb)

        data_for_vdb = {
            compute_mdhash_id(chunk_id + dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "description": add_period_if_needed(dp["description"]),
                "content": dp["keywords"] + " " + dp["src_id"] + " " + dp["tgt_id"] + " "
                           + add_period_if_needed(dp["description"]),
            } for chunk_id in relations_kv_all for dp in relations_kv_all[chunk_id][0]
        }
        await self.relations_vdb.upsert(data_for_vdb)

        data_for_vdb = {
            chunk_id: {
                "content": chunks_summary_kv_all[chunk_id]["summary"]
            } for chunk_id in chunks_summary_kv_all
        }
        await self.chunks_summary_vdb.upsert(data_for_vdb)

        # insert in knowledge graph
        for chunk_id in chunks_kv_all:
            for entity in entities_kv_all[chunk_id][0]:
                entity['description'] = add_period_if_needed(entity['description'])
                await _merge_nodes_then_upsert(entity["entity_name"], [entity], self.chunk_entity_relation_graph)

        for chunk_id in chunks_kv_all:
            for relation in relations_kv_all[chunk_id][0]:
                relation['description'] = add_period_if_needed(relation['description'])
                await _merge_edges_then_upsert(relation["src_id"], relation["tgt_id"], [relation],
                                               self.chunk_entity_relation_graph)

        await self._insert_done()

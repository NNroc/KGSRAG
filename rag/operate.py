import ast
import asyncio
import copy
import json
import re
import warnings
import tiktoken
import numpy as np
import torch
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    process_combine_all_contexts,
    process_combine_statements,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData, cal_list_token,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS


def deduplicate_list_of_dicts(list_of_dicts: list):
    seen = set()
    unique_list = []
    for d in list_of_dicts:
        d_frozenset = d['description']
        if d_frozenset not in seen:
            seen.add(d_frozenset)
            unique_list.append(d)
    return unique_list


def chunking_by_token_size(content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=tiktoken_model,
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    texts = text_splitter.create_documents([content])
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    token_counts = [len(encoder.encode(t.page_content)) for t in texts]
    for index, tokens in enumerate(texts):
        results.append(
            {
                "tokens": token_counts[index],
                "content": tokens.page_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


def unique_data(data):
    seen = set()
    unique_data = []
    for item in data:
        if item['source_id'] not in seen:
            seen.add(item['source_id'])
            unique_data.append(item)
    return unique_data


async def _handle_entity_relation_summary(
        entity_or_relation_name: str,
        description: str,
        global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(tokens[:llm_max_tokens], model_name=tiktoken_model_name)
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
        record_attributes: list[str],
        chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relation_extraction(record_attributes: list[str], chunk_key: str):
    if len(record_attributes) < 5 or record_attributes[0] != '"relation"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0)
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
        entity_name: str,
        nodes_data: list[dict],
        knowledge_graph_inst: BaseGraphStorage,
):
    already_entity_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.extend(split_string_by_multi_markers(already_node["entity_type"], [GRAPH_FIELD_SEP]))
        already_source_ids.extend(split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP]))
        already_description.extend(split_string_by_multi_markers(already_node["description"], [GRAPH_FIELD_SEP]))

    entity_type = GRAPH_FIELD_SEP.join(set([dp["entity_type"] for dp in nodes_data] + already_entity_types))
    source_id = GRAPH_FIELD_SEP.join(set([dp["source_id"] for dp in nodes_data] + already_source_ids))
    description = GRAPH_FIELD_SEP.join(sorted(set([dp["description"] for dp in nodes_data] + already_description)))
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(entity_name, node_data=node_data)
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
        src_id: str,
        tgt_id: str,
        edges_data: list[dict],
        knowledge_graph_inst: BaseGraphStorage,
):
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_source_ids.extend(split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP]))
        already_description.extend(split_string_by_multi_markers(already_edge["description"], [GRAPH_FIELD_SEP]))
        already_keywords.extend(split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP]))

    source_id = GRAPH_FIELD_SEP.join(set([dp["source_id"] for dp in edges_data] + already_source_ids))
    description = GRAPH_FIELD_SEP.join(sorted(set([dp["description"] for dp in edges_data] + already_description)))
    keywords = GRAPH_FIELD_SEP.join(sorted(set([dp["keywords"] for dp in edges_data] + already_keywords)))
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities_only(
        chunks: dict[str, TextChunkSchema],
        chunks_kv: BaseKVStorage,
        entities_kv: BaseKVStorage,
        relations_kv: BaseKVStorage,
        global_config: dict,
):
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES_MEDICINE"])
    entity_extract_prompt = PROMPTS["entity_extraction"]
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    # custom
    if global_config["dataset"] == "custom":
        entity_types = global_config["addon_params"].get("entity_types", PROMPTS["entity_type_custom"])
        examples = "\n".join(PROMPTS["entity_extraction_examples_custom"])
        entity_extract_prompt = PROMPTS["entity_extraction_custom"]

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types="[" + ",".join(entity_types) + "]",
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types="[" + ",".join(entity_types) + "]",
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        reie = True
        reie_context = ""
        reie_num = 0
        while reie and reie_num < 3:
            hint_prompt = entity_extract_prompt.format(
                **context_base, input_text="{input_text}").format(**context_base, input_text=content + reie_context)
            final_result = await use_llm_func(hint_prompt, max_new_tokens=20480)
            history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
            for now_glean_index in range(entity_extract_max_gleaning):
                glean_result = await use_llm_func(continue_prompt, history_messages=history, max_new_tokens=20480)

                history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
                final_result += glean_result
                if now_glean_index == entity_extract_max_gleaning - 1:
                    break

                if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break

            records = split_string_by_multi_markers(
                final_result, [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )

            maybe_nodes = defaultdict(list)
            maybe_edges = defaultdict(list)
            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])
                if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key)
                if if_entities is not None:
                    maybe_nodes[if_entities["entity_name"]].append(if_entities)
                    continue

                if_relation = await _handle_single_relation_extraction(record_attributes, chunk_key)
                if if_relation is not None:
                    maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)

            if context_base["tuple_delimiter"] not in final_result:
                reie = True
                reie_context = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
                reie_num += 1
            else:
                reie = False
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="", flush=True, )
        return dict(maybe_nodes), dict(maybe_edges)

    extraction_num = 0
    for result in tqdm_async(ordered_chunks, total=len(ordered_chunks),
                             desc="Extracting entities from chunks", unit="chunk"):
        # 我需要在这里进行判断，该kvdb中是否存储
        if await chunks_kv.get_by_id(result[0]):
            continue
        all_entities, all_relations = await _process_single_content(result)
        entities_set = set()
        for key, value_list in all_entities.items():
            for value in value_list:
                # 将键和字典转换为元组并添加到集合中
                entities_set.add((key, tuple(value.items())))
        # 将集合中的元组转换回字典列表
        entities = [dict(item[1]) for item in entities_set]

        relations_set = set()
        for key, value_list in all_relations.items():
            for value in value_list:
                # 将键和字典转换为元组并添加到集合中
                relations_set.add((key, tuple(value.items())))
        # 将集合中的元组转换回字典列表
        relations = [dict(item[1]) for item in relations_set]

        await chunks_kv.upsert({result[0]: {"content": result[1]["content"]}})
        await entities_kv.upsert({result[0]: [entities]})
        await relations_kv.upsert({result[0]: [relations]})

    await chunks_kv.index_done_callback()
    await entities_kv.index_done_callback()
    await relations_kv.index_done_callback()


async def texts_summary(
        chunks: dict[str, TextChunkSchema],
        chunks_summary_kv: BaseKVStorage,
        global_config: dict,
):
    use_llm_func: callable = global_config["llm_model_func"]
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    example_number = global_config["addon_params"].get("example_number", None)
    texts_summary_prompt = PROMPTS["text_summary"]
    texts_summary_example = PROMPTS["text_summary_examples"]
    examples = "\n".join(texts_summary_example)
    ordered_chunks = list(chunks.items())

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        repeat = True
        repeat_num = 0
        while repeat and repeat_num < 10:
            hint_prompt = texts_summary_prompt.format(examples=examples, input_text=content, language=language)
            final_result = await use_llm_func(hint_prompt)
            pattern = r'"summary": "([^"]*)"'
            summary_matches = re.findall(pattern, final_result)
            try:
                text_summary = summary_matches[0]
                return text_summary
            except Exception as e:
                repeat_num += 1
                continue
        return ""

    for result in tqdm_async(ordered_chunks, total=len(ordered_chunks),
                             desc="Extracting entities from chunks", unit="chunk"):
        # 我需要在这里进行判断，该kvdb中是否存储
        if await chunks_summary_kv.get_by_id(result[0]):
            continue

        summary = await _process_single_content(result)
        await chunks_summary_kv.upsert({
            result[0]: {"summary": summary,
                        "content": result[1]['content']}
        })
        await chunks_summary_kv.index_done_callback()


async def extract_entities(
        chunks: dict[str, TextChunkSchema],
        chunks_vdb: BaseKVStorage,
        entities_vdb: BaseKVStorage,
        relations_vdb: BaseKVStorage,
        merge_entities_vdb: BaseVectorStorage,
        merge_relations_vdb: BaseVectorStorage,
        global_config: dict,
        knowledge_graph_inst: BaseGraphStorage
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES_MEDICINE"])
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
        examples = "\n".join(PROMPTS["entity_extraction_examples"][: int(example_number)])
    else:
        examples = "\n".join(PROMPTS["entity_extraction_examples"])

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types="[" + ",".join(entity_types) + "]",
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types="[" + ",".join(entity_types) + "]",
        examples=examples,
        language=language,
    )

    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await use_llm_func(hint_prompt)
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result, [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"]])
            if_entities = await _handle_single_entity_extraction(record_attributes, chunk_key)
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relation_extraction(record_attributes, chunk_key)
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(if_relation)
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    results = []
    extraction_num = 0
    for result in tqdm_async(ordered_chunks, total=len(ordered_chunks),
                             desc="Extracting entities from chunks", unit="chunk"):
        # 我需要在这里进行判断，该vdb中是否存储
        if await chunks_vdb.get_by_id(result[0]):
            # a=chunks_vdb.get_by_id(result[0])
            ### results.append()
            continue
        all_entities, all_relations = await _process_single_content(result)
        entities_set = set()
        for key, value_list in all_entities.items():
            for value in value_list:
                # 将键和字典转换为元组并添加到集合中
                entities_set.add((key, tuple(value.items())))
        # 将集合中的元组转换回字典列表
        entities = [dict(item[1]) for item in entities_set]

        relations_set = set()
        for key, value_list in all_relations.items():
            for value in value_list:
                # 将键和字典转换为元组并添加到集合中
                relations_set.add((key, tuple(value.items())))
        # 将集合中的元组转换回字典列表
        relations = [dict(item[1]) for item in relations_set]

        ### results.append()

        await chunks_vdb.upsert({result[0]: {"content": result[1]["content"]}})
        await entities_vdb.upsert({result[0]: [entities]})
        await relations_vdb.upsert({result[0]: [relations]})

        extraction_num += 1
        if extraction_num % 100 == 0:
            await chunks_vdb.index_done_callback()
            await entities_vdb.index_done_callback()
            await relations_vdb.index_done_callback()

    await chunks_vdb.index_done_callback()
    await entities_vdb.index_done_callback()
    await relations_vdb.index_done_callback()

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)
    logger.info("Inserting entities into storage...")
    all_entities_data = []
    for result in tqdm_async(
            asyncio.as_completed(
                [
                    _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                    for k, v in maybe_nodes.items()
                ]
            ),
            total=len(maybe_nodes),
            desc="Inserting entities",
            unit="entity",
    ):
        all_entities_data.append(await result)

    logger.info("Inserting relations into storage...")
    all_relations_data = []
    for result in tqdm_async(
            asyncio.as_completed(
                [
                    _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
                    for k, v in maybe_edges.items()
                ]
            ),
            total=len(maybe_edges),
            desc="Inserting relations",
            unit="relation",
    ):
        all_relations_data.append(await result)

    if not len(all_entities_data) and not len(all_relations_data):
        logger.warning("Didn't extract any entities and relations, maybe your LLM is not working")
        return None

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relations_data):
        logger.warning("Didn't extract any relations")

    if merge_entities_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await merge_entities_vdb.upsert(data_for_vdb)

    if merge_relations_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"] + dp["src_id"] + dp["tgt_id"] + dp["description"],
            }
            for dp in all_relations_data
        }
        await merge_relations_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def kg_query(
        query, query_list, high_list, lower_list, answer_list,
        knowledge_graph_inst: BaseGraphStorage,
        chunks_vdb: BaseVectorStorage,
        chunks_summary_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage,
        relations_vdb: BaseVectorStorage,
        chunks_kv: BaseKVStorage,
        entities_kv: BaseKVStorage,
        relations_kv: BaseKVStorage,
        query_param: QueryParam,
        global_config: dict,
        query_type=None,
        hashing_kv: BaseKVStorage = None,
        cal_tokens=False
):
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.decomposition, query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(hashing_kv, args_hash, query, query_param.mode)
    if cached_response is not None:
        return cached_response
    if len(query_list) == 0:
        query_list = [query, "UNKNOW"]
    # Set mode
    if query_param.mode not in ["naive", "perfect", "keyword", "statement", "all"]:
        logger.error(f"Unknown mode {query_param.mode} in kg_query")
        return PROMPTS["fail_response"]

    # Build context
    keywords = [high_list, lower_list]
    context = await _build_query_context(
        query, query_list, knowledge_graph_inst, keywords,
        chunks_vdb, chunks_summary_vdb, entities_vdb, relations_vdb,
        chunks_kv, entities_kv, relations_kv,
        query_param, )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    if cal_tokens:
        tokens_num = len(encode_string_by_tiktoken(context))
        return tokens_num

    if global_config["dataset"] is not None:
        if global_config["dataset"] == "bioasq":
            sys_prompt_temp = PROMPTS["rag_response_" + global_config["dataset"] + "_" + query_type]
        else:
            sys_prompt_temp = PROMPTS["rag_response_" + global_config["dataset"]]
    else:
        sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(context_data=context)
    if query_param.only_need_prompt:
        return sys_prompt
    response = await use_model_func(query, system_prompt=sys_prompt, stream=query_param.stream)
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )
    return response


async def _build_query_context(
        query: str,
        statements: list,
        knowledge_graph_inst: BaseGraphStorage,
        keywords: list,
        chunks_vdb: BaseVectorStorage,
        chunks_summary_vdb: BaseVectorStorage,
        entities_vdb: BaseVectorStorage,
        relations_vdb: BaseVectorStorage,
        chunks_kv: BaseKVStorage,
        entities_kv: BaseKVStorage,
        relations_kv: BaseKVStorage,
        query_param: QueryParam,
):
    ll_entities_context, ll_relations_context, ll_text_units_context = "", "", ""
    hl_entities_context, hl_relations_context, hl_text_units_context = "", "", ""
    s_entities_context, s_relations_context, s_text_units_context = "", "", ""
    entities_context, relations_context, text_units_context, knowledge_statements = [], [], [], ""
    statements_embeddings = None
    ll_keywords, hl_keywords = keywords[0], keywords[1]

    if query_param.mode == 'perfect':
        data_qa = json.load(open(f'data/{query_param.dataset}_qa.json', 'r'))
        new_docs = []
        for data in data_qa:
            if data['question'] == query:
                context = data['contexts']
                new_docs = {
                    compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
                    for c in context
                }
                break
        chunks_all_data = await chunks_kv.all_data()
        need_chunks_ids = []
        need_chunks = []
        for chunk_ids in chunks_all_data:
            if chunks_all_data[chunk_ids]['full_doc_id'] in new_docs:
                need_chunks_ids.append(chunk_ids)
                need_chunks.append(chunks_all_data[chunk_ids]['content'])

        entities = await entities_kv.get_by_ids(need_chunks_ids)
        relations = await relations_kv.get_by_ids(need_chunks_ids)
        use_entities, use_relations = [], []
        for e in entities:
            use_entities.extend(e[0])
        for r in relations:
            use_relations.extend(r[0])
        entities_section_list = [["id", "entity", "type", "description"]]
        for i, e in enumerate(use_entities):
            entities_section_list.append(
                [
                    i,
                    e["entity_name"],
                    e["entity_type"],
                    e["description"],
                ]
            )
        entities_context = list_of_list_to_csv(entities_section_list)
        relations_section_list = [["id", "source", "target", "description", "keywords"]]
        for i, e in enumerate(use_relations):
            relations_section_list.append(
                [
                    i,
                    e["src_id"],
                    e["tgt_id"],
                    e["description"],
                    e["keywords"],
                ]
            )
        relations_context = list_of_list_to_csv(relations_section_list)
        text_units_section_list = [["id", "content"]]
        for i, t in enumerate(need_chunks):
            text_units_section_list.append([i, t])
        text_units_context = list_of_list_to_csv(text_units_section_list)

    #     if query_param.mode == 'all':
    #         statements_section_list = [["id", "sub query", "sub query response"]]
    #         for i, e in enumerate(statements):
    #             statements_section_list.append([i, e[0], e[1]])
    #         statements_context = list_of_list_to_csv(statements_section_list)
    #         knowledge_statements = f"""
    # -----Statements-----
    # ```csv
    # {statements_context}
    # ```
    # """
    # if len(statements) > 0 and (query_param.mode == 'statement' or query_param.mode == 'all'):
    #     s_entities_context, s_relations_context, s_text_units_context = \
    #         await _get_chunk_data(query, statements, chunks_vdb, chunks_summary_vdb, knowledge_graph_inst,
    #                               chunks_kv, entities_kv, relations_kv, query_param)
    #     s_entities_context, s_relations_context, s_text_units_context = "", "", ""
    if query_param.mode == "all":
        statements_embeddings = await entities_vdb.embedding(statements)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        statements_embeddings_np = np.array([se[0] for se in statements_embeddings])
        statements_embeddings = torch.tensor(statements_embeddings_np, device=device)
    if len(ll_keywords) > 0 and (query_param.mode == 'keyword' or query_param.mode == 'all'):
        ll_entities_context, ll_relations_context, ll_text_units_context = \
            await _get_node_data(ll_keywords, statements, entities_vdb, relations_vdb, chunks_kv, entities_kv,
                                 knowledge_graph_inst, query_param, statements_embeddings, 1.5)
    if len(hl_keywords) > 0 and (query_param.mode == 'keyword' or query_param.mode == 'all'):
        hl_entities_context, hl_relations_context, hl_text_units_context, = \
            await _get_edge_data(hl_keywords, statements, entities_vdb, relations_vdb, chunks_kv, relations_kv,
                                 knowledge_graph_inst, query_param, statements_embeddings, 1.5)
    if query_param.mode == "all":
        entities_context, relations_context, text_units_context = combine_all_contexts(
            [hl_entities_context, ll_entities_context, s_entities_context],
            [hl_relations_context, ll_relations_context, s_relations_context],
            [hl_text_units_context, ll_text_units_context, s_text_units_context],
        )
    elif query_param.mode == "statement":
        entities_context, relations_context, text_units_context = combine_all_contexts(
            [s_entities_context],
            [s_relations_context],
            [s_text_units_context],
        )
    elif (query_param.mode == "keyword") and len(ll_keywords) > 0 and len(hl_keywords) > 0:
        entities_context, relations_context, text_units_context = combine_all_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )
    elif (query_param.mode == "keyword") and len(ll_keywords) > 0:
        entities_context, relations_context, text_units_context = (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        )
    elif (query_param.mode == "keyword") and len(hl_keywords) > 0:
        entities_context, relations_context, text_units_context = (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        )
    knowledge = f"""
-----Entities-----
```csv
{entities_context}
```
-----Relations-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```"""

    # knowledge = f"""
    # -----Sources-----
    # ```csv
    # {text_units_context}
    # ```
    # """

    return knowledge


async def _get_chunk_data(
        query,
        statements,
        chunks_vdb: BaseVectorStorage,
        chunks_summary_vdb: BaseVectorStorage,
        knowledge_graph_inst: BaseGraphStorage,
        chunks_kv: BaseKVStorage,
        entities_kv: BaseKVStorage,
        relations_kv: BaseKVStorage,
        query_param: QueryParam,
        top_k=10
):
    qp = copy.deepcopy(query_param)
    qp.top_k = top_k
    qp.dynamic_threshold = False
    results = []
    for statement in statements:
        results.extend(await chunks_summary_vdb.query(statement[0], qp))
    results_unique = dict()
    for res in results:
        if res['__metrics__'] < 0.40:
            continue
        if res['__id__'] not in results_unique:
            results_unique[res['__id__']] = res['__metrics__']
        else:
            results_unique[res['__id__']] = max(res['__metrics__'], results_unique[res['__id__']])
    use_results = []
    for key, value in results_unique.items():
        use_results.append({'__id__': key, '__metrics__': value})

    use_results.sort(key=lambda x: (x['__metrics__'], x['__id__']), reverse=True)

    if not len(use_results):
        return "", "", ""

    chunks_ids = [r["__id__"] for r in use_results]
    chunks = await chunks_kv.get_by_ids(chunks_ids)
    # Filter out invalid chunks
    valid_chunks = [chunk for chunk in chunks if chunk is not None and "content" in chunk]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return PROMPTS["fail_response"]

    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        return PROMPTS["fail_response"]

    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")

    entities = await entities_kv.get_by_ids(chunks_ids)
    relations = await relations_kv.get_by_ids(chunks_ids)

    entities_data, relations_data = [], []
    for e in entities:
        entities_data.extend(e[0])
    for r in relations:
        relations_data.extend(r[0])

    # get entity information
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entities_data]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    # get entity degree
    node_degrees = await asyncio.gather(*[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entities_data])
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(entities_data, node_datas, node_degrees)
        if n is not None
    ]

    # get relation information
    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in relations_data]
    )
    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in relations_data])
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(relations_data, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(edge_datas, key=lambda x: (x["rank"]), reverse=True)
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    # 去重
    node_datas_deduplication = deduplicate_list_of_dicts(node_datas)
    edge_datas_deduplication = deduplicate_list_of_dicts(edge_datas)
    # node_datas_deduplication = node_datas_deduplication[:query_param.top_k]
    # edge_datas_deduplication = edge_datas_deduplication[:query_param.top_k]

    use_entities, use_relations, use_text_units = node_datas_deduplication, edge_datas_deduplication, valid_chunks
    use_entities.extend(await _find_most_related_entities_from_relations(edge_datas_deduplication, query_param,
                                                                         knowledge_graph_inst))
    u_relations = await _find_most_related_edges_from_entities(node_datas_deduplication, query_param,
                                                               knowledge_graph_inst)
    for u in u_relations:
        use_relations.append({
            'src_id': u['src_tgt'][0],
            'tgt_id': u['src_tgt'][1],
            'rank': u['rank'],
            # 'description': u['description'],
            'description': '',
            'keywords': u['keywords'],
            'source_id': u['source_id']
        })

    # use_text_units.extend(await _find_most_related_text_unit_from_entities(
    #     node_datas_deduplication, query_param, chunks_kv, knowledge_graph_inst))
    # use_text_units.extend(await _find_related_text_unit_from_relations(
    #     edge_datas_deduplication, query_param, chunks_kv, knowledge_graph_inst))

    entities_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, e in enumerate(entities_data):
        entities_section_list.append(
            [
                i,
                e["entity_name"],
                e["entity_type"],
                # e["description"],
                '',
                "",
            ]
        )
    entities_context = list_of_list_to_csv(entities_section_list)

    relations_section_list = [["id", "source", "target", "description", "keywords", "rank"]]
    for i, e in enumerate(relations_data):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                # e["description"],
                '',
                e["keywords"],
                "",
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return entities_context, relations_context, text_units_context


async def _get_node_data(
        query,
        statements,
        entities_vdb: BaseVectorStorage,
        relations_vdb: BaseVectorStorage,
        text_chunks_kv: BaseKVStorage[TextChunkSchema],
        entities_kv: BaseKVStorage,
        knowledge_graph_inst: BaseGraphStorage,
        query_param: QueryParam,
        statements_embeddings,
        magnification=1.5
):
    # get similar entities
    qp = copy.deepcopy(query_param)
    results = await entities_vdb.query(query, qp)

    if not len(results):
        return "", "", ""

    # get entity information
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    # get entity degree
    node_degrees = await asyncio.gather(*[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results])
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_kv doing.  dont remember it in arxiv.  check the diagram.
    use_entities = copy.deepcopy(node_datas)
    # get entity text chunk
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_kv, knowledge_graph_inst
    )
    # get relate edges
    use_relations = await _find_most_related_edges_from_entities(node_datas, query_param, knowledge_graph_inst)
    logger.info(
        f"Nodes query uses {len(node_datas)} entities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    use_entities = unique_data(use_entities)
    use_relations = unique_data(use_relations)

    if query_param.mode == "all":
        use_filter_entities = await entities_vdb.filter(use_entities, statements_embeddings, 0.4)
        use_filter_relations = await relations_vdb.filter(use_relations, statements_embeddings, 0.2)
        # use_filter_text_units = await entities_vdb.filter(results, statements)
        use_entities = use_filter_entities
        use_relations = use_filter_relations

    # build prompt
    entities_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entities_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entities_section_list)
    relations_section_list = [["id", "source", "target", "description", "keywords", "rank"]]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_text_unit_from_entities(
        node_datas: list[dict],
        query_param: QueryParam,
        text_chunks_db: BaseKVStorage[TextChunkSchema],
        knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(*[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas])
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (e[1] in all_one_hop_text_units_lookup and c_id in all_one_hop_text_units_lookup[e[1]]):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(all_text_units, key=lambda x: (x["order"], -x["relation_counts"]))

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
        node_datas: list[dict],
        query_param: QueryParam,
        knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack = await asyncio.gather(*[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges])
    all_edges_degree = await asyncio.gather(*[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges])
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(all_edges_data, key=lambda x: (x["rank"]), reverse=True)
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def _get_edge_data(
        keywords,
        statements,
        entities_vdb: BaseVectorStorage,
        relations_vdb: BaseVectorStorage,
        text_chunks_kv: BaseKVStorage[TextChunkSchema],
        relations_kv: BaseKVStorage,
        knowledge_graph_inst: BaseGraphStorage,
        query_param: QueryParam,
        statements_embeddings,
        magnification=1.5
):
    qp = copy.deepcopy(query_param)
    results = await relations_vdb.query(keywords, qp)

    if not len(results):
        return "", "", ""

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(*[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results])
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(edge_datas, key=lambda x: (x["rank"]), reverse=True)
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    use_relations = copy.deepcopy(edge_datas)
    use_entities = await _find_most_related_entities_from_relations(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relations(
        edge_datas, query_param, text_chunks_kv, knowledge_graph_inst
    )
    logger.info(
        f"Edges query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    use_entities = unique_data(use_entities)
    use_relations = unique_data(use_relations)

    if query_param.mode == "all":
        use_filter_entities = await entities_vdb.filter(use_entities, statements_embeddings, 0.4)
        use_filter_relations = await relations_vdb.filter(use_relations, statements_embeddings, 0.2)
        use_entities = use_filter_entities
        use_relations = use_filter_relations

    relations_section_list = [["id", "source", "target", "description", "keywords", "rank"]]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entities_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entities_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entities_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context


async def _find_most_related_entities_from_relations(
        edge_datas: list[dict],
        query_param: QueryParam,
        knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names])

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relations(
        edge_datas: list[dict],
        query_param: QueryParam,
        text_chunks_db: BaseKVStorage[TextChunkSchema],
        knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]
    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                chunk_data = await text_chunks_db.get_by_id(c_id)
                # Only store valid data
                if chunk_data is not None and "content" in chunk_data:
                    all_text_units_lookup[c_id] = {
                        "data": chunk_data,
                        "order": index,
                    }

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [t for t in all_text_units if t["data"] is not None and "content" in t["data"]]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    truncated_text_units = truncate_list_by_token_size(
        valid_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units


def combine_contexts(entities, relations, sources):
    # Function to extract entities, relations, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relations, ll_relations = relations[0], relations[1]
    hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    # Combine and deduplicate the relations
    combined_relations = process_combine_contexts(hl_relations, ll_relations)
    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)
    return combined_entities, combined_relations, combined_sources


def combine_all_contexts(entities, relations, sources):
    # Combine and deduplicate the entities
    combined_entities = process_combine_all_contexts(entities)
    # Combine and deduplicate the relations
    combined_relations = process_combine_all_contexts(relations)
    # Combine and deduplicate the sources
    combined_sources = process_combine_all_contexts(sources)
    return combined_entities, combined_relations, combined_sources


async def naive_query(
        query,
        chunks_vdb: BaseVectorStorage,
        chunks_kv: BaseKVStorage[TextChunkSchema],
        query_param: QueryParam,
        global_config: dict,
        query_type: None,
        hashing_kv: BaseKVStorage = None,
        cal_tokens=False
):
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(hashing_kv, args_hash, query, query_param.mode)
    if cached_response is not None:
        return cached_response

    qp = copy.deepcopy(query_param)
    qp.dynamic_threshold = False
    results = await chunks_vdb.query(query, qp)
    if not len(results):
        print(PROMPTS["fail_response"])
        # return PROMPTS["fail_response"]

    chunks_ids = [r["id"] for r in results]
    chunks = await chunks_kv.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [chunk for chunk in chunks if chunk is not None and "content" in chunk]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        # return PROMPTS["fail_response"]

    maybe_trun_chunks, tokens_num = cal_list_token(
        valid_chunks,
        key=lambda x: x["content"]
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        # return PROMPTS["fail_response"]

    if cal_tokens:
        return tokens_num

    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "\n--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])

    if query_param.only_need_context:
        return section

    # sys_prompt_temp = PROMPTS["naive_rag_response"]
    if global_config["dataset"] is not None:
        if global_config["dataset"] == "bioasq":
            sys_prompt_temp = PROMPTS["rag_response_" + global_config["dataset"] + "_" + query_type]
        else:
            sys_prompt_temp = PROMPTS["rag_response_" + global_config["dataset"]]
    else:
        sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(context_data=section)

    if query_param.only_need_prompt:
        return sys_prompt

    response = await use_model_func(query, system_prompt=sys_prompt)

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt):]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
        ),
    )

    return response


async def llm_decomposition(question,
                            question_param: QueryParam,
                            question_vdb: BaseKVStorage,
                            global_config: dict) -> tuple[list, list, list, list]:
    use_llm_func: callable = global_config["llm_model_func"]
    # add language and example number params to prompt
    language = global_config["addon_params"].get("language", PROMPTS["DEFAULT_LANGUAGE"])
    entity_types = global_config["addon_params"].get("entity_types", PROMPTS["DEFAULT_ENTITY_TYPES_MEDICINE"])
    example_number = global_config["addon_params"].get("example_number", None)
    if question_param.decomposition == "statement":
        decomposition_prompt = PROMPTS["question_decomposition"]
        decomposition_example = PROMPTS["question_decomposition_examples"]
    elif question_param.decomposition == "question":
        decomposition_prompt = PROMPTS["question_decomposition_question"]
        decomposition_example = PROMPTS["question_decomposition_question_examples"]
    if example_number and example_number < len(PROMPTS["question_decomposition_examples"]):
        examples = "\n".join(decomposition_example[: int(example_number)])
    else:
        examples = "\n".join(decomposition_example)

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types="[" + ",".join(entity_types) + "]",
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types="[" + ",".join(entity_types) + "]",
        examples=examples,
        language=language,
    )

    question_decomposition_prompt = decomposition_prompt
    new_questions = {compute_mdhash_id(question.strip(), prefix="que-"): {"content": question.strip()}}
    new_questions = list(new_questions.items())
    new_questions = new_questions[0]

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        reqd = True
        reqd_context = ""
        reqd_num = 0
        maybe_queries_response = defaultdict(list)
        maybe_final_response, maybe_high, maybe_lower = [], [], []
        while reqd and reqd_num < 10:
            hint_prompt = question_decomposition_prompt.format(
                **context_base, input_text="{input_text}" + reqd_context
            ).format(**context_base, input_text=content)
            final_result = await use_llm_func(hint_prompt)
            try:
                delimiter = re.findall(r'"sub"(.*?)"', final_result)[0]
            except:
                delimiter = context_base["tuple_delimiter"]
            records = split_string_by_multi_markers(
                final_result, [context_base["record_delimiter"], context_base["completion_delimiter"]])
            maybe_queries_response = defaultdict(list)
            maybe_final_response, maybe_high, maybe_lower = [], [], []
            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(record, [context_base["tuple_delimiter"], delimiter])
                if record_attributes[0] == '"sub"' and len(record_attributes) >= 3:
                    maybe_queries_response[record_attributes[1]].append(record_attributes[2])
                    continue
                if record_attributes[0] == '"response"' and len(record_attributes) >= 2:
                    maybe_final_response.append(record_attributes[1])
                    continue
                if record_attributes[0] == '"high"' and len(record_attributes) >= 2:
                    for ra in record_attributes[1:]:
                        maybe_high.append(ra)
                if record_attributes[0] == '"lower"' and len(record_attributes) >= 2:
                    for ra in record_attributes[1:]:
                        maybe_lower.append(ra)
            if len(maybe_queries_response) == 0 or len(maybe_final_response) == 0:
                reqd = True
                reqd_context = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
                reqd_num += 1
            else:
                reqd = False

        return dict(maybe_queries_response), maybe_final_response, maybe_high, maybe_lower

    # 我需要在这里进行判断，该vdb中是否存储
    if await question_vdb.get_by_id(new_questions[0]):
        result = await question_vdb.get_by_id(new_questions[0])
        return result["sub"], result["high"], result["lower"], result["response"]
    all_queries_response, all_final_response, all_high, all_lower = await _process_single_content(new_questions)
    queries_set = set()
    for key, value_list in all_queries_response.items():
        for value in value_list:
            # 将键和字典转换为元组并添加到集合中
            queries_set.add((key, value))
    # 将集合中的元组转换回字典列表
    queries_list = [[item[0], item[1]] for item in queries_set]
    await question_vdb.upsert(
        {new_questions[0]: {"question": new_questions[1]["content"],
                            "response": all_final_response, "sub": queries_list,
                            "high": all_high, "lower": all_lower}
         })
    await question_vdb.index_done_callback()
    return queries_list, all_high, all_lower, all_final_response

import networkx as nx
import sys
import json


def inspect_graphml(path: str, output_path: str = None):
    G = nx.read_graphml(path)

    if output_path is None:
        output_path = path.replace(".graphml", "_dump.json")

    nodes = {}
    for node_id, data in G.nodes(data=True):
        nodes[node_id] = dict(data)

    edges = []
    for src, tgt, data in G.edges(data=True):
        edge = {"src": src, "tgt": tgt}
        edge.update(data)
        edges.append(edge)

    dump = {
        "file": path,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_directed": G.is_directed(),
        "nodes": nodes,
        "edges": edges,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)

    print(f"Dumped {len(nodes)} nodes, {len(edges)} edges -> {output_path}")
    return dump


def filter_by_source_id(dump: dict, chunk_id: str, output_path: str = None):
    _drop_keys = ("source_id", "description")
    nodes = {
        nid: {k: v for k, v in data.items() if k not in _drop_keys}
        for nid, data in dump["nodes"].items()
        if chunk_id in str(data.get("source_id", ""))
    }
    edges = []
    for e in dump["edges"]:
        if chunk_id in str(e.get("source_id", "")):
            edge = {k: v for k, v in e.items() if k not in _drop_keys}
            edges.append(edge)
    filtered = {
        "chunk_id": chunk_id,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }

    if output_path is None:
        output_path = dump.get("file", "").replace(".graphml", "") + f"_filtered_{chunk_id}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Filtered: {len(nodes)} nodes, {len(edges)} edges -> {output_path}")
    return filtered


if __name__ == "__main__":
    default_path = "/data2/npy/projects/KGSRAG/output/bioasq_Qwen2.5-14B-Instruct-2/graph_chunk_entity_relation.graphml"
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    output = sys.argv[2] if len(sys.argv) > 2 else None

    dump = inspect_graphml(path, output)

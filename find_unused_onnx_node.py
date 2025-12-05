import onnx
from collections import defaultdict, deque

def find_unused_nodes(model_path: str):
    model = onnx.load(model_path)
    g = model.graph

    producer = {}
    consumers = defaultdict(list)

    nodes = list(g.node)
    # ノード → インデックスの対応を作る
    node_index = {id(n): i for i, n in enumerate(nodes)}

    for node in nodes:
        for out_name in node.output:
            producer[out_name] = node
        for in_name in node.input:
            consumers[in_name].append(node)

    reachable_node_ids = set()
    reachable_values = set()

    q = deque()
    for out in g.output:
        if out.name:
            q.append(out.name)
            reachable_values.add(out.name)

    while q:
        v = q.popleft()
        node = producer.get(v)
        if node is None:
            continue
        nid = id(node)
        if nid in reachable_node_ids:
            continue
        reachable_node_ids.add(nid)
        for in_name in node.input:
            if in_name and in_name not in reachable_values:
                reachable_values.add(in_name)
                q.append(in_name)

    unused_nodes = [n for n in nodes if id(n) not in reachable_node_ids]

    print(f"total nodes      : {len(nodes)}")
    print(f"reachable nodes  : {len(reachable_node_ids)}")
    print(f"unused nodes     : {len(unused_nodes)}")

    for n in unused_nodes:
        print("UNUSED:", n.op_type, n.name, "->", list(n.output))

    return unused_nodes

unused = find_unused_nodes("tmp/dynamo_const_nodyn_fix.onnx")

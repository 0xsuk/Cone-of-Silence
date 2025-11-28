import json
from collections import defaultdict
import argparse

def analyze_profile(path, top=30):
    with open(path, "r") as f:
        data = json.load(f)

    # node ごとの合計時間 (dur は us 単位)
    by_node = defaultdict(int)
    by_op   = defaultdict(int)

    for e in data:
        if e.get("cat") != "Node":
            continue
        args = e.get("args") or {}
        op_type   = args.get("op_name") or args.get("op_type")
        node_name = args.get("node_name") or e.get("name")
        dur = e.get("dur", 0)  # microseconds

        if node_name:
            by_node[node_name] += dur
        if op_type:
            by_op[op_type] += dur

    print("=== Top nodes ===")
    for name, dur in sorted(by_node.items(), key=lambda x: x[1], reverse=True)[:top]:
        print(f"{dur/1000:.3f} ms\t{name}")

    print("\n=== Time by op type ===")
    for op, dur in sorted(by_op.items(), key=lambda x: x[1], reverse=True):
        print(f"{dur/1000:.3f} ms\t{op}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="onnxruntime_profile_....json")
    parser.add_argument("--top", type=int, default=30, help="show top N nodes")
    args = parser.parse_args()

    analyze_profile(args.json_path, args.top)

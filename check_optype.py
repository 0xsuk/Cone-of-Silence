import onnx

model = onnx.load("tmp/pdynamo_const_nodyn21.onnx")

op_types = {node.op_type for node in model.graph.node}

print(op_types)

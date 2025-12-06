import onnx

model = onnx.load("tmp/qpdynamo_const_nodyn_fix_noconv2.onnx")

op_types = {node.op_type for node in model.graph.node}

print(op_types)

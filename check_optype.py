import onnx

model = onnx.load("./tmp/dq_sqpdynamo_const_nodyn21_u_qo.onnx")

op_types = {node.op_type for node in model.graph.node}

print(op_types)

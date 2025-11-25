import onnx
model = onnx.load("model_quant.onnx")
for node in model.graph.node:
    if node.op_type == "DynamicQuantizeLSTM":
        print("Found DynamicQuantizeLSTM:", node)
        print("Attributes:", node.attribute)
        

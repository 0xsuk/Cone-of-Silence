import onnxruntime as ort

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = "tmp/model_opset18_opt.onnx"

sess = ort.InferenceSession("tmp/model_opset18.onnx", sess_options=so, providers=["CPUExecutionProvider"])

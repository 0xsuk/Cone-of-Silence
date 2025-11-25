from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'tmp/model_opset21_const.onnx'
model_quant = 'tmp/qmodel_opset21_const_int.onnx'
quantize_dynamic(model_fp32, model_quant,weight_type=QuantType.QInt8, )

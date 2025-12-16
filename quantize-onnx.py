from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'tmp/sqpdynamo_const_nodyn21_u__convonly.onnx'
model_quant = 'tmp/dq_sqpdynamo_const_nodyn21_u__convonly.onnx'
quantize_dynamic(model_fp32, model_quant,weight_type=QuantType.QInt8,
                 op_types_to_quantize=["LSTM", "MatMul"] #IntegerOpsRegistryの中から選ぶ. GEMMをquantizeするならMatmulをいれる
                 )
#activationはuint8しかサポートされていない

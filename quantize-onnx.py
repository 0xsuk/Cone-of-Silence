from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'tmp/pdynamo_const_nodyn_fix.onnx'
model_quant = 'tmp/qpdynamo_const_nodyn_fix.onnx'
quantize_dynamic(model_fp32, model_quant,weight_type=QuantType.QInt8,
                 # op_types_to_quantize=['Reshape', 'Gemm', 'Tanh', 'Slice', 'Mul', 'Sigmoid', 'Concat', 'Transpose', 'Squeeze', 'ConvTranspose', 'Relu', 'Split', 'Add'],
                 # op_types_to_quantize=["LSTM", "Attention", "MatMul", "Conv"] #IntegerOpsRegistryの中から選ぶ. GEMMをquantizeするならMatmulをいれる
                 )
#activationはuint8しかサポートされていない

import onnxruntime as ort

# 利用可能なすべてのExecution Providerのリストを取得
available_providers = ort.get_available_providers()

print(available_providers)

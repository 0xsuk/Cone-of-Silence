import onnx
from onnxscript import optimizer as ox_opt

model = onnx.load("./tmp/dynamo_const_noopt_opset.onnx")  # 最適化前 or 最小限のグラフ
opt_model = ox_opt.optimize(model)    # ここで同じ AttributeError が出るかを見る

import torch, torch.nn as nn
from cos.training.network import CoSNetwork


# 1) モデル定義を用意してインスタンス化
model = CoSNetwork()
state = torch.load("checkpoints/realdata_4mics_.03231m_44100kHz.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()

# 2) 動的量子化（x86はfbgemm、ARMはqnnpackを想定）
torch.backends.quantized.engine = "qnnpack"  # or "|fbgemm"
qmodel = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.LSTM}, dtype=torch.qint8,
    reduce_range=False #raspiでfalseじゃないと[W qlinear_dynamic.cpp:239] Warning: Currently, qnnpack incorrectly ignores reduce_range when it is set to true; this may change in a future release. (function apply_dynamic_impl)

)

# 3) 保存と推論
torch.save(qmodel, "model_int8_dyn_arm_model.pt")

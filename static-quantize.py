import torch
import torch.nn as nn
from cos.training.network import CoSNetwork

# モデル読み込み
model_fp32 = CoSNetwork()
state = torch.load("checkpoints/realdata_4mics_.03231m_44100kHz.pt", map_location="cpu")
model_fp32.load_state_dict(state)
model_fp32.eval()

# 量子化設定（ARMならqnnpack）
torch.backends.quantized.engine = "qnnpack"

# 1. 準備: observer挿入
model_fp32.qconfig = torch.quantization.get_default_qconfig("qnnpack")
model_prepared = torch.quantization.prepare(model_fp32)

n_classes = 5  # 例: window_conditioning_size
for _ in range(50):
    mix = torch.randn(1, 4, 44100)
    # one-hot ラベルをランダム生成
    idx = torch.randint(0, n_classes, (1,))
    angle_conditioning = torch.zeros(1, n_classes)
    angle_conditioning[0, idx] = 1.0
    model_prepared(mix, angle_conditioning)

# 3. 変換: 実際に量子化
model_int8 = torch.quantization.convert(model_prepared)

# 4. 保存
torch.save(model_int8, "model_int8_static_arm.pt")

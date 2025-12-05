import torch
from cos.training.network import CoSNetwork

def load_state_dict_safely(ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state

# モデルの初期化とウェイトのロード
model = CoSNetwork()
ckpt_path = "checkpoints/realdata_4mics_.03231m_44100kHz.pt"
state = load_state_dict_safely(ckpt_path)
model.load_state_dict(state, strict=True)
model.eval()

# ここでtorch.compileを適用する
# modeは 'default', 'reduce-overhead', 'max-autotune' などから選択
compiled_model = torch.compile(model, fullgraph=True)

# 以降の推論には compiled_model を使用する
# ダミー入力（B, C, T）と条件ラベル（B, COND_DIM）
COND_DIM = 5
DUMMY_T = 44100

dummy_audio = torch.randn(1, 4, DUMMY_T, dtype=torch.float32)
dummy_cond  = torch.zeros(1, COND_DIM, dtype=torch.float32)
dummy_cond[:, 0] = 1.0

# 初回実行時（コンパイルが行われる）
output = compiled_model(dummy_audio, dummy_cond)

# 2回目以降の実行は高速化される

import argparse
import torch
import numpy as np
from cos.training.network import CoSNetwork

# --- 必要に応じて変更 ---
OPSET = 17          # 16〜21 あたりで安定。エラー時は 16/18 も試す
COND_DIM = 5        # conditioning の one-hot 次元（コードでは 5）
DUMMY_T = 44100     # ダミーの時間長（動的軸にするので適当でOK）
# -----------------------

def load_state_dict_safely(ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state

@torch.no_grad()
def export_onnx(ckpt_path: str, onnx_path: str, n_channels: int):
    model = CoSNetwork(n_audio_channels=n_channels)
    model.load_state_dict(load_state_dict_safely(ckpt_path), strict=True)
    model.eval()  # 必須
    model.cpu()   # ONNX Export は CPU で

    # ダミー入力（B, C, T）と条件ラベル（B, COND_DIM）
    dummy_audio = torch.randn(1, n_channels, DUMMY_T, dtype=torch.float32)
    dummy_cond  = torch.zeros(1, COND_DIM, dtype=torch.float32)
    dummy_cond[:, 0] = 1.0  # 適当な one-hot

    # 動的軸を設定：バッチと時間長を可変に
    dynamic_axes = {
        "audio": {0: "batch", 2: "time"},
        "cond":  {0: "batch"},
        "output": {0: "batch", 2: "time"}  # モデル出力が (B, C, T) を想定
    }

    torch.onnx.export(
        model,
        (dummy_audio, dummy_cond),
        onnx_path,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,   # もし最適化で問題が出たら False に
        input_names=["audio", "cond"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    print(f"Exported ONNX to: {onnx_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_checkpoint", type=str)
    p.add_argument("onnx_out", type=str)
    p.add_argument("--n_channels", type=int, default=2)
    args = p.parse_args()

    export_onnx(args.model_checkpoint, args.onnx_out, args.n_channels)

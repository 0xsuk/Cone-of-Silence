import argparse
import torch
from cos.training.network import CoSNetwork


OPSET = 23
COND_DIM = 5
DUMMY_T = 144724 # 3秒のオーディオ(44100*3 + valid_length() によるpadding)

class CoSNetworkFixedCond(torch.nn.Module):
    def __init__(self, base: CoSNetwork, fixed_cond: torch.Tensor):
        super().__init__()
        self.base = base
        # (1, COND_DIM) を register_buffer しておくと ONNX 上は initializer になる
        self.register_buffer("fixed_cond", fixed_cond.clone())

    def forward(self, mix: torch.Tensor):
        # バッチサイズに合わせて繰り返す
        cond = self.fixed_cond.expand(mix.size(0), -1)
        return self.base(mix, cond)


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
    dummy_cond[:, 0] = 1.0  # 90度を指定

    wrapped = CoSNetworkFixedCond(model, dummy_cond)

    
    torch.onnx.export(
        wrapped,
        (dummy_audio,),
        onnx_path,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["audio"],
        output_names=["output"],
        # dynamic_axes=dynamic_axes,
        optimize=True, #default is True, only valid when dynamo=True
        dynamo=True
    )
    print(f"Exported ONNX to: {onnx_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_checkpoint", type=str)
    p.add_argument("onnx_out", type=str)
    p.add_argument("--n_channels", type=int, default=4)
    args = p.parse_args()

    export_onnx(args.model_checkpoint, args.onnx_out, args.n_channels)

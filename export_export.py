import argparse
import torch
from cos.training.network import CoSNetwork

OPSET = 24
COND_DIM = 5
DUMMY_T = 44100


def load_state_dict_safely(ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def dump_exported_program_graph(model: torch.nn.Module,
                                dummy_audio: torch.Tensor,
                                dummy_cond: torch.Tensor) -> None:
    """
    torch.export.export で ExportedProgram を取り、そのグラフをダンプする。
    Dynamo ベース ONNX exporter が見る IR を先に確認するためのデバッグ用。
    """
    # CoSNetwork.forward のシグネチャに合わせて args を渡す
    exported = torch.export.export(
        model,
        (dummy_audio, dummy_cond),
        dynamic_shapes=None,   # 必要ならここに dynamic shape 指定を書く
    )
    print("=== ExportedProgram graph ===")
    # テキストでざっくり眺めたい場合
    print(exported.graph)

    # FX GraphModule としても見たい場合（お好みで）
    gm = exported.module()
    print("=== FX GraphModule (print_tabular) ===")
    try:
        gm.graph.print_tabular()
    except Exception:
        # 古い PyTorch などで print_tabular が無い場合
        print(gm.graph)


@torch.no_grad()
def export_onnx(ckpt_path: str, onnx_path: str, n_channels: int):
    model = CoSNetwork(n_audio_channels=n_channels)
    model.load_state_dict(load_state_dict_safely(ckpt_path), strict=True)
    model.eval()
    model.cpu()

    # ダミー入力（B, C, T）と条件ラベル（B, COND_DIM）
    dummy_audio = torch.randn(1, n_channels, DUMMY_T, dtype=torch.float32)
    dummy_cond = torch.zeros(1, COND_DIM, dtype=torch.float32)
    dummy_cond[:, 0] = 1.0

    # まず Dynamo/torch.export がどんな IR を作っているか見る
    dump_exported_program_graph(model, dummy_audio, dummy_cond)

    # 動的軸を設定：バッチと時間長を可変に
    dynamic_axes = {
        "audio": {0: "batch", 2: "time_in"},
        "cond": {0: "batch"},
        "output": {0: "batch", 3: "time_out"},  # モデル出力が (B,2, C, T) を想定
    }

    # 実際の ONNX export（optimizer をオフ）
    # torch.onnx.export(
    #     model,
    #     (dummy_audio, dummy_cond),
    #     onnx_path,
    #     export_params=True,
    #     opset_version=OPSET,
    #     do_constant_folding=True,
    #     input_names=["audio", "cond"],
    #     output_names=["output"],
    #     dynamic_axes=dynamic_axes,
    #     optimize=False,  # default True, only valid when dynamo=True
    #     dynamo=True,
    # )
    print(f"Exported ONNX to: {onnx_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model_checkpoint", type=str)
    p.add_argument("onnx_out", type=str)
    p.add_argument("--n_channels", type=int, default=4)
    args = p.parse_args()

    export_onnx(args.model_checkpoint, args.onnx_out, args.n_channels)

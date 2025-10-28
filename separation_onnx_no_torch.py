import argparse, os
from collections import namedtuple
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort

from cos.helpers.constants import ALL_WINDOW_SIZES, FAR_FIELD_RADIUS
from cos.helpers.visualization import draw_diagram
from cos.helpers.eval_utils import si_sdr

# ---- 定数（元実装準拠） ----
ENERGY_CUTOFF = 0.0000
NMS_RADIUS = np.pi / 4
NMS_SIMILARITY_SDR = -7.0
COND_DIM = 5
SPEED_OF_SOUND = 343.0  # m/s

CandidateVoice = namedtuple("CandidateVoice", ["angle", "energy", "data"])

# ---- 便利関数（NumPy版） ----
def to_categorical(i, classes):
    one = np.zeros((classes,), dtype=np.float32)
    one[i] = 1.0
    return one

def center_trim_np(x, ref_len):
    """ x: (C, T) or (T,) """
    if x.ndim == 1:
        T = x.shape[0]
        if T == ref_len: return x
        if T > ref_len:
            delta = T - ref_len
            return x[delta//2: delta//2 + ref_len]
        else:
            pad = ref_len - T
            return np.pad(x, (pad//2, pad - pad//2))
    else:
        C, T = x.shape
        if T == ref_len: return x
        if T > ref_len:
            d = T - ref_len
            return x[:, d//2: d//2 + ref_len]
        else:
            p = ref_len - T
            return np.pad(x, ((0,0),(p//2, p - p//2)))

def normalize_input_np(x):
    """ x: (1, C, T) -> (1, C, T), means (1,C,1), stds (1,C,1) """
    means = x.mean(axis=2, keepdims=True)
    stds = x.std(axis=2, keepdims=True) + 1e-8
    return (x - means) / stds, means, stds

def unnormalize_input_np(x, means, stds):
    return x * stds + means

def get_mic_array(n_channels, radius):
    """等角度に円周配置（x-y 平面, mic0=角度0）"""
    ang = np.arange(n_channels, dtype=np.float32) * (2*np.pi / n_channels)
    xy = np.stack([radius*np.cos(ang), radius*np.sin(ang)], axis=1)  # (C,2)
    return xy

def fractional_delay_sig(sig, delay_sec, sr):
    """sig: (T,), delay_sec: float -> (T,)  小数サンプル遅延（sinc 補間）"""
    sig = np.asarray(sig).reshape(-1)
    if abs(delay_sec) < 1e-12:
        return sig
    T = sig.shape[0]
    n = np.arange(T)
    frac = delay_sec * sr
    # 窓付きsinc（有限長）
    L = 32
    k = np.arange(-L, L+1)
    h = np.sinc(k - frac) * np.hamming(2*L+1)
    h = h / (h.sum() + 1e-12)
    y = np.convolve(sig, h, mode='same')
    return y

def shift_mixture_np(audio, target_pos_xy, mic_radius, sr):
    """
    audio: (C, T) float32
    target_pos_xy: np.array([x,y], float32)
    -> (C, T) （candidate 方向に整列させる：各chに相対遅延を適用）
    """
    C, T = audio.shape
    mic_xy = get_mic_array(C, mic_radius)                 # (C,2)
    # 距離差 → 遅延 [sec]
    dists = np.linalg.norm(mic_xy - target_pos_xy[None,:], axis=1)  # (C,)
    # mic0 基準
    delays = (dists - dists[0]) / SPEED_OF_SOUND                       # (C,)
    out = np.zeros_like(audio)
    for c in range(C):
        out[c] = fractional_delay_sig(audio[c], -delays[c], sr)  # アライン方向に負符号
    return out

def angular_distance(a, b):
    """最小角度差 [0,pi]"""
    d = np.abs(a-b) % (2*np.pi)
    return np.minimum(d, 2*np.pi - d)

def nms(candidate_voices, nms_cutoff):
    final_proposals = []
    initial = candidate_voices
    while len(initial) > 0:
        new_initial = []
        sorted_c = sorted(initial, key=lambda x: x[1], reverse=True)
        best = sorted_c[0]
        final_proposals.append(best)
        sorted_c.pop(0)
        for cand in sorted_c:
            different_locations = angular_distance(cand.angle, best.angle) > NMS_RADIUS
            different_content = si_sdr(cand.data[0], best.data[0]) < nms_cutoff
            if different_locations or different_content:
                new_initial.append(cand)
        initial = new_initial
    return final_proposals

# ---- ORT 実行 & valid_length 推定（Torch不要） ----
def ort_try_run(session, audio_1ct, cond_1d):
    """ audio_1ct: (1,C,T), cond_1d: (1,COND_DIM) -> 出力 (1,C,T') """
    inputs = session.get_inputs()
    feed = {
        inputs[0].name: audio_1ct.astype(np.float32),
        inputs[1].name: cond_1d.astype(np.float32),
    }
    return session.run(None, feed)[0]

def find_valid_length(session, C, T, cond_1d):
    """与えた T を中心パディングで増やしつつ、モデルが通る長さを探す"""
    # よくある 2**k 倍にスナップ（上限を決めて試行）
    for mul in [32, 64, 128, 256, 512, 1024]:
        vlen = int(np.ceil(T / mul) * mul)
        pad_l = (vlen - T)//2
        pad_r = vlen - T - pad_l
        test = np.zeros((1,C,T), np.float32)
        test = np.pad(test, ((0,0),(0,0),(pad_l, pad_r)))
        try:
            _ = ort_try_run(session, test, cond_1d)
            return vlen, pad_l, pad_r
        except Exception:
            continue
    # フォールバックで 1 サンプルずつ増やす（最大 +4096）
    for add in range(0, 4097):
        vlen = T + add
        pad_l = add//2
        pad_r = add - pad_l
        test = np.zeros((1,C,T), np.float32)
        test = np.pad(test, ((0,0),(0,0),(pad_l, pad_r)))
        try:
            _ = ort_try_run(session, test, cond_1d)
            return vlen, pad_l, pad_r
        except Exception:
            pass
    raise RuntimeError("valid_length が見つかりませんでした。")

# ---- 1 ステップ ----
def forward_pass_onnx(session, target_angle, mixed_data_ct, cond_1d, sr, mic_radius):
    """
    mixed_data_ct: (C,T) float32
    cond_1d: (1,COND_DIM)
    """
    target_pos = np.array([
        FAR_FIELD_RADIUS*np.cos(target_angle),
        FAR_FIELD_RADIUS*np.sin(target_angle)
    ], dtype=np.float32)

    # シフト（NumPy）
    data_ct = shift_mixture_np(mixed_data_ct, target_pos, mic_radius, sr)  # (C,T)

    # 1バッチ化
    data_1ct = data_ct[None, ...]  # (1,C,T)
    # 正規化
    data_norm, means, stds = normalize_input_np(data_1ct)

    # valid_length を探索し、中心パディング
    B, C, T = data_norm.shape
    vlen, pad_l, pad_r = find_valid_length(session, C, T, cond_1d)
    padded = np.pad(data_norm, ((0,0),(0,0),(pad_l, pad_r)))

    # ORT 実行
    ort_out = ort_try_run(session, padded, cond_1d)  # (1,C,T')
    # center_trim to original T
    out_1ct = ort_out
    out_1ct = out_1ct[:, :, (out_1ct.shape[2] - T)//2 : (out_1ct.shape[2] - T)//2 + T]
    # 逆正規化
    out_1ct = unnormalize_input_np(out_1ct, means, stds)
    out_first = out_1ct[:, 0, :]  # (1,T)

    output_np = out_first[0]      # (T,)
    energy = float(librosa.feature.rms(y=output_np).mean())
    return output_np, energy

def run_separation(mixed_data_ct, session, sr, mic_radius,
                   energy_cutoff=ENERGY_CUTOFF, nms_cutoff=NMS_SIMILARITY_SDR,
                   moving=False, debug=False, writing_dir=None):
    num_windows = 3 if moving else len(ALL_WINDOW_SIZES)
    starting_angles = np.asarray(np.linspace(0, 2*np.pi, num=8, endpoint=False), dtype=np.float32)  # 既存utils同等の初期角がなければ代替
    starting_angles = starting_angles[-2:]  # 元コードの変更に合わせる：45度と135度

    candidate_voices = [CandidateVoice(x, None, None) for x in starting_angles]
    print("candidate", candidate_voices)
    print("starting angles", starting_angles)

    for window_idx in range(num_windows):
        if debug: print("---------")
        cond = to_categorical(window_idx, COND_DIM)[None, :]  # (1,5)

        curr_window_size = ALL_WINDOW_SIZES[window_idx]
        print("window: ", curr_window_size)
        new_candidate_voices = []

        for voice in candidate_voices:
            output, energy = forward_pass_onnx(session, voice.angle, mixed_data_ct, cond, sr, mic_radius)

            if debug and writing_dir:
                print(f"Angle {voice.angle:.2f} energy {energy}")
                fname = f"out{window_idx}_angle{voice.angle * 180 / np.pi:.2f}.wav"
                sf.write(os.path.join(writing_dir, fname), output, sr)

            if energy > energy_cutoff:
                if window_idx == num_windows - 1:
                    # 逆シフト（元の幾何に戻す）: forward の逆方向
                    target_pos = np.array([
                        FAR_FIELD_RADIUS*np.cos(voice.angle),
                        FAR_FIELD_RADIUS*np.sin(voice.angle)
                    ], dtype=np.float32)
                    # forward で「整列のために -delay」を当てたので、逆シフトは +delay
                    # shift_mixture_np は -delay を当てる実装なので、逆方向にするため target_pos を同じにしつつ out を再シフトしないように…はできない。
                    # ここでは「正方向 delay」を与えるバージョンを適用：
                    C = mixed_data_ct.shape[0]
                    mic_xy = get_mic_array(C, mic_radius)
                    d0 = np.linalg.norm(mic_xy - target_pos[None,:], axis=1)
                    delays = (d0 - d0[0]) / SPEED_OF_SOUND  # 正方向
                    out_1d = np.asarray(output).reshape(-1)           # ← 念のため 1D 化
                    Tlen = int(out_1d.shape[0])
                    restored = np.zeros((C, Tlen), dtype=np.float32)
                    for c in range(C):
                        restored[c, :] = fractional_delay_sig(out_1d, delays[c], sr)
                else:
                    new_candidate_voices.append(CandidateVoice(voice.angle + curr_window_size / 4, energy, None))
                    new_candidate_voices.append(CandidateVoice(voice.angle - curr_window_size / 4, energy, None))

        candidate_voices = new_candidate_voices

    return nms(candidate_voices, nms_cutoff)

def main(args):
    print("use cuda", args.use_cuda)
    shouldSave = args.save

    providers = ["CPUExecutionProvider"] if not args.use_cuda else ["CUDAExecutionProvider","CPUExecutionProvider"]
    try:
        ort_sess = ort.InferenceSession(args.model_onnx, providers=providers)
    except Exception:
        ort_sess = ort.InferenceSession(args.model_onnx, providers=["CPUExecutionProvider"])

    if not os.path.exists(args.output_dir) and shouldSave:
        os.makedirs(args.output_dir)

    mixed = librosa.core.load(args.input_file, mono=False, sr=args.sr)[0].astype(np.float32)  # (C,T)
    assert mixed.shape[0] == args.n_channels

    temporal_chunk_size = int(args.sr * args.duration)
    num_chunks = (mixed.shape[1] // temporal_chunk_size) + 1

    for chunk_idx in range(num_chunks):
        curr_dir = os.path.join(args.output_dir, f"{chunk_idx:03d}")
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
        curr = mixed[:, (chunk_idx*temporal_chunk_size): (chunk_idx+1)*temporal_chunk_size]

        outputs = run_separation(curr, ort_sess, args.sr, args.mic_radius,
                                 moving=args.moving, debug=args.debug, writing_dir=curr_dir)

        if shouldSave:
            print("save", outputs)
            for v in outputs:
                fname = "output_angle{:.2f}.wav".format(v.angle * 180 / np.pi)
                sf.write(os.path.join(curr_dir, fname), v.data[0], args.sr)
            candidate_angles = [v.angle for v in outputs]
            diagram_window_angle = ALL_WINDOW_SIZES[2] if args.moving else ALL_WINDOW_SIZES[-1]
            draw_diagram([], candidate_angles, diagram_window_angle, os.path.join(curr_dir, "positions.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_onnx", type=str)
    ap.add_argument("input_file", type=str)
    ap.add_argument("output_dir", type=str)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_channels", type=int, default=2)
    ap.add_argument("--use_cuda", dest="use_cuda", action="store_true")
    ap.add_argument("--save", dest="save", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--mic_radius", default=.03231, type=float)
    ap.add_argument("--duration", default=3.0, type=float)
    ap.add_argument("--moving", action="store_true")
    main(ap.parse_args())

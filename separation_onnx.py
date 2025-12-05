"""
ONNXRuntime version of: separation_by_localization.py
- 前処理/後処理/valid_length は Python 側（Torch/NumPy）
- モデル本体の forward は ONNXRuntime
"""

import argparse
import os
from collections import namedtuple

import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
import json
import torch
import torch.nn.functional as F
import time

import cos.helpers.utils as utils
from cos.helpers.constants import ALL_WINDOW_SIZES, FAR_FIELD_RADIUS
from cos.helpers.visualization import draw_diagram
from cos.training.network import CoSNetwork, center_trim, normalize_input, unnormalize_input
from cos.helpers.eval_utils import si_sdr

# Constants
ENERGY_CUTOFF = 0.00001
NMS_RADIUS = np.pi / 4
NMS_SIMILARITY_SDR = -7.0  # SDR cutoff for different candidates
COND_DIM = 5               # one-hot の次元（元コード準拠）

CandidateVoice = namedtuple("CandidateVoice", ["angle", "energy", "data"])


def _make_session(onnx_path: str, use_cuda: bool):
    
    # providers = [("ACLExecutionProvider", {"enable_fast_math": "true"})]
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    # CUDA が未インストールでも落ちないようフォールバック
    so = ort.SessionOptions()
    # so.enable_profiling = True
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(onnx_path, so, providers=providers)
        print("session providers:", sess.get_providers())
        return sess
    except Exception as e:
        raise e;


def nms(candidate_voices, nms_cutoff):
    final_proposals = []
    initial_proposals = candidate_voices

    while len(initial_proposals) > 0:
        new_initial_proposals = []
        sorted_candidates = sorted(initial_proposals, key=lambda x: x[1], reverse=True)

        best_candidate_voice = sorted_candidates[0]
        final_proposals.append(best_candidate_voice)
        sorted_candidates.pop(0)

        for candidate_voice in sorted_candidates:
            different_locations = utils.angular_distance(candidate_voice.angle, best_candidate_voice.angle) > NMS_RADIUS
            different_content = si_sdr(candidate_voice.data[0], best_candidate_voice.data[0]) < nms_cutoff
            if different_locations or different_content:
                new_initial_proposals.append(candidate_voice)

        initial_proposals = new_initial_proposals

    return final_proposals


def process_voice(voice, ort_sess, valid_length_fn, mixed_data, conditioning_label, args,
                  window_idx, energy_cutoff, num_windows, curr_window_size):
    output, energy = forward_pass_onnx(ort_sess, valid_length_fn, voice.angle, mixed_data,
                                  conditioning_label, args)
    results = []

    if args.debug:
        print(f"Angle {voice.angle:.2f} energy {energy}")
        fname = f"out{window_idx}_angle{voice.angle * 180 / np.pi:.2f}.wav"
        sf.write(os.path.join(args.writing_dir, fname), output[0], args.sr)

    if energy > energy_cutoff:
        if window_idx == num_windows - 1:
            target_pos = np.array([
                FAR_FIELD_RADIUS * np.cos(voice.angle),
                FAR_FIELD_RADIUS * np.sin(voice.angle)
            ])
            unshifted_output, _ = utils.shift_mixture(
                output, target_pos, args.mic_radius, args.sr, inverse=True
            )
            results.append(CandidateVoice(voice.angle, energy,
                                          unshifted_output))
        else:
            results.append(
                CandidateVoice(voice.angle + curr_window_size / 4, energy,
                               output)
            )
            results.append(
                CandidateVoice(voice.angle - curr_window_size / 4, energy,
                               output)
            )
    return results

from concurrent.futures import ThreadPoolExecutor,as_completed
def separate(candidate_voices, ort_sess, valid_length_fn, mixed_data, conditioning_label, args,
             window_idx, energy_cutoff, num_windows, new_candidate_voices,
             curr_window_size, parallel=False):
    
    if parallel:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_voice,
                    voice, ort_sess, valid_length_fn, mixed_data, conditioning_label, args,
                    window_idx, energy_cutoff, num_windows, curr_window_size
                )
                for voice in candidate_voices
            ]
            for future in as_completed(futures):
                new_candidate_voices.extend(future.result())
    else:
        for voice in candidate_voices:
            result = process_voice(
                voice, ort_sess, valid_length_fn, mixed_data, conditioning_label, args,
                window_idx, energy_cutoff, num_windows, curr_window_size
            )
            new_candidate_voices.extend(result)

def forward_pass_onnx(ort_sess, valid_length_fn, target_angle, mixed_data, conditioning_label_onehot, args):
    """
    - mixed_data: np.ndarray, shape (C, T)
    - conditioning_label_onehot: np.ndarray, shape (1, COND_DIM)
    """
    target_pos = np.array([
        FAR_FIELD_RADIUS * np.cos(target_angle),
        FAR_FIELD_RADIUS * np.sin(target_angle)
    ], dtype=np.float32)

    # shift_mixture は torch 実装なので一旦 torch.Tensor にして実行（CPUでOK）
    data_t, _ = utils.shift_mixture(
        torch.tensor(mixed_data), target_pos,
        args.mic_radius, args.sr
    )
    data_t = data_t.float().unsqueeze(0)  # (1, C, T)

    # Normalize
    data_t, means, stds = normalize_input(data_t)  # tensors

    # valid_length とパディング（元実装と同一ロジック）
    T = data_t.shape[-1]
    print("before :", data_t.shape)
    vlen = valid_length_fn(T)
    delta = int(vlen - T)
    padded_t = F.pad(data_t, (delta // 2, delta - delta // 2))  # (1, C, vlen)
    print("after  :", padded_t.shape)

    # ORT 実行
    audio_np = padded_t.detach().cpu().numpy().astype(np.float32)  # (1, C, vlen)
    cond_np = conditioning_label_onehot.astype(np.float32)         # (1, COND_DIM)

    ort_inputs = {ort_sess.get_inputs()[0].name: audio_np,
                  # ort_sess.get_inputs()[1].name: cond_np
                  }
    ort_out = ort_sess.run(None, ort_inputs)[0]  # (1, C, T')

    # center_trim/unnormalize は torch 実装に合わせる
    out_t = torch.from_numpy(ort_out)
    out_t = center_trim(out_t, data_t)          # (1, C, T)
    out_t = unnormalize_input(out_t, means, stds)
    out_first_mic = out_t[:, 0]                 # (1, T)

    output_np = out_first_mic.detach().cpu().numpy()[0]  # (T,)
    energy = float(librosa.feature.rms(y=output_np).mean())

    return output_np, energy


def run_separation(mixed_data, ort_sess, valid_length_fn, args,
                   energy_cutoff=ENERGY_CUTOFF,
                   nms_cutoff=NMS_SIMILARITY_SDR):
    """
    The main separation by localization algorithm (ORT 版)
    """
    num_windows = len(ALL_WINDOW_SIZES) if not args.moving else 3
    starting_angles = utils.get_starting_angles(ALL_WINDOW_SIZES[0])
    starting_angles = starting_angles[-2:]  # 45度と135度

    candidate_voices = [CandidateVoice(x, None, None) for x in starting_angles]
    print("candidate", candidate_voices)
    print("starting angles", starting_angles)

    start = time.time()
    for window_idx in range(num_windows):
        if args.debug:
            print("---------")

        # one-hot conditioning (1, COND_DIM)
        cond = utils.to_categorical(window_idx, COND_DIM).astype(np.float32)[None, :]

        curr_window_size = ALL_WINDOW_SIZES[window_idx]
        print("window: ", curr_window_size)
        new_candidate_voices = []

        separate(candidate_voices=candidate_voices,
                 ort_sess=ort_sess,
                 valid_length_fn=valid_length_fn,
                 mixed_data=mixed_data,
                 conditioning_label=cond,
                 args=args,
                 window_idx=window_idx,
                 energy_cutoff=energy_cutoff,
                 num_windows=num_windows,
                 new_candidate_voices=new_candidate_voices,
                 curr_window_size=curr_window_size
                 )
        candidate_voices = new_candidate_voices

    end = time.time()
    print(f"run_separation: {end - start:.4f} seconds")
    return nms(candidate_voices, nms_cutoff)




def profile(sess):
    profile_file = sess.end_profiling()
    print("profile file:", profile_file)
    
    # JSON を読み込んで、どの EP がどれくらい時間を使ったか集計する
    with open(profile_file, "r") as f:
        data = json.load(f)
    
    # event 内に "args" や "cat" 等で EP 名や op 名が入っているので、それを集計
    ep_time = {}
    for e in data:
        if "args" not in e:
            continue
        args = e["args"]
        provider = args.get("provider")
        dur = e.get("dur", 0)  # us 単位
        if provider:
            ep_time[provider] = ep_time.get(provider, 0) + dur
    
    for ep, dur in ep_time.items():
        print(ep, dur / 1000.0, "ms")    

def main(args):
    print("use cuda", args.use_cuda)
    shouldSave = args.save

    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    args.device = device

    
    # ONNX セッション
    ort_sess = _make_session(args.model_onnx, use_cuda=args.use_cuda)

    # valid_length 計算のためだけに CoSNetwork を生成（重みロード不要）
    _shape_helper = CoSNetwork(n_audio_channels=args.n_channels)
    valid_length_fn = _shape_helper.valid_length

    if not os.path.exists(args.output_dir) and shouldSave:
        os.makedirs(args.output_dir)

    mixed_data = librosa.core.load(args.input_file, mono=False, sr=args.sr)[0]  # (C, T)
    assert mixed_data.shape[0] == args.n_channels, f"n_channels mismatch: file={mixed_data.shape[0]}, arg={args.n_channels}"

    temporal_chunk_size = int(args.sr * args.duration)
    num_chunks = (mixed_data.shape[1] // temporal_chunk_size) + 1

    for chunk_idx in range(num_chunks):
        curr_writing_dir = os.path.join(args.output_dir, f"{chunk_idx:03d}")
        if not os.path.exists(curr_writing_dir):
            os.makedirs(curr_writing_dir)

        args.writing_dir = curr_writing_dir
        curr_mixed_data = mixed_data[:, (chunk_idx * temporal_chunk_size): (chunk_idx + 1) * temporal_chunk_size]

        output_voices = run_separation(curr_mixed_data, ort_sess, valid_length_fn, args)

        if shouldSave:
            for voice in output_voices:
                fname = "output_angle{:.2f}.wav".format(voice.angle * 180 / np.pi)
                # voice.data は (1, T) を想定
                sf.write(os.path.join(args.writing_dir, fname), voice.data[0], args.sr)

            candidate_angles = [voice.angle for voice in output_voices]
            diagram_window_angle = ALL_WINDOW_SIZES[2] if args.moving else ALL_WINDOW_SIZES[-1]
            draw_diagram([], candidate_angles, diagram_window_angle,
                         os.path.join(args.writing_dir, "positions.png"))


    # profile(ort_sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_onnx', type=str, help="Path to the ONNX model file")
    parser.add_argument('input_file', type=str, help="Path to the input file (wav)")
    parser.add_argument('output_dir', type=str, help="Path to write the outputs")
    parser.add_argument('--sr', type=int, default=22050, help="Sampling rate")
    parser.add_argument('--n_channels', type=int, default=2, help="Number of channels")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true', help="Use CUDAExecutionProvider if available")
    parser.add_argument('--save', dest='save', action='store_true', help="save output and draw image")
    parser.add_argument('--debug', action='store_true', help="Save intermediate outputs")
    parser.add_argument('--mic_radius', default=.03231, type=float, help="Radius of the mic array")
    parser.add_argument('--duration', default=3.0, type=float, help="Seconds of input to the network")
    parser.add_argument('--moving', action='store_true', help="If sources are moving then stop at a coarse window")
    main(parser.parse_args())

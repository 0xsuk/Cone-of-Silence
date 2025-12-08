import librosa
import numpy as np
import torch
import torch.nn.functional as F

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
)
from onnxruntime.quantization import quant_utils
from pathlib import Path
from cos.training.network import CoSNetwork, normalize_input

SR = 44100
N_CHANNELS = 4
COND_DIM = 5
DURATION = 3.0


class SingleAudioDataReader(CalibrationDataReader):
    def __init__(self, wav_path: str, onnx_model_path: str):
        self.wav_path = wav_path
        self._consumed = False

        
        # valid_length 用
        shape_helper = CoSNetwork(n_audio_channels=N_CHANNELS)
        self.valid_length_fn = shape_helper.valid_length

        # 入力名を取得
        model = quant_utils.load_model_with_shape_infer(Path(onnx_model_path))

        self.input_name = model.graph.input[0].name

    def get_next(self):
        if self._consumed:
            return None
        self._consumed = True

        wav, sr = librosa.core.load(self.wav_path, mono=False, sr=SR)
        if wav.ndim == 1:
            wav = wav[np.newaxis, :]
        assert wav.shape[0] == N_CHANNELS

        temporal_chunk_size = int(SR * DURATION)
        wav = wav[:, :temporal_chunk_size]

        data_t = torch.tensor(wav).float().unsqueeze(0)  # (1, C, T)

        data_t, means, stds = normalize_input(data_t)
        T = data_t.shape[-1]
        vlen = self.valid_length_fn(T)
        delta = int(vlen - T)
        padded_t = F.pad(data_t, (delta // 2, delta - delta // 2))  # (1, C, vlen)

        audio_np = padded_t.detach().cpu().numpy().astype(np.float32)
        return {self.input_name: audio_np}

    def rewind(self):
        self._consumed = False


def static_quantize_single_file(
):
    model_fp32_path = "tmp/pdynamo_const_nodyn21.onnx"
    model_int8_path = "tmp/sqpdynamo_const_nodyn21_u_qo.onnx"
    dr = SingleAudioDataReader("real_multiple_speakers_4mics.wav", model_fp32_path)

    quantize_static(
        model_input=model_fp32_path,
        model_output=model_int8_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8, #
        weight_type=QuantType.QInt8,
        # per_channel=True,
        # op_types_to_quantize=["Conv", "MatMul"],
    )
    print("saved:", model_int8_path)


static_quantize_single_file()

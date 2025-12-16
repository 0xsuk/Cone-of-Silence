# cone of silence 使い方など

## 背景
このリポジトリはCone-of-silenceのforkで、開発元のコードのバグ修正や古いパッケージのアップロードをしています。

pytorchの学習済みモデル(checkpoints/realdata_4mics_.03231m_44100kHz.pt)が開発元のreadmeのリンクからダウンロードできるので、それをつかってonnxへの変換や量子化を行います


## インストール
```
uv venv
source .venv/bin/activate

uv pip install myrequirements.txt
```

myrequirements.txtはfork用のpip freeze出力です。

```
torch 2.9.1+cpu
onnx 1.19.1
onnxruntime: 1.24.0
```
であればよく他のパッケージのバージョンは重要ではないです。


## 使い方
checkpoints/realdata_4mics_.03231m_44100kHz.ptを用意 <- readme
### pytorch -> onnx
```
python export_onnx_sane.py checkpoints/realdata_4mics_.03231m_44100kHz.pt model.onnx
```

### onnx量子化
推論を高速にするためにonnxモデルのint8量子化を行えます.

#### 動的量子化

quantize-onnx.pyのmodel_fp32, model_quantのファイルパスをセットしたあと
```
python quantize-onnx.py
```
で量子化します。  
quantize-onnx.py内のop_types_to_quantizeには[IntegerOpsRegistry](https://github.com/microsoft/onnxruntime/blob/ff815674fdbbe6b1b78309950c2dad9d49cf4e8f/onnxruntime/python/tools/quantization/registry.py#L32)のキーを指定します。

動的量子化はLSTMやMatMul,(Cone-of-silenceでは使われていないがAttention)に対して効果的で、Convに対してはほとんど高速化が期待できません。その理屈はまだ理解していません。  


#### 静的量子化
同様にstatic-quantize-onnx.py内のファイルパスを編集した後
```
python static-quantize-onnx.py
```
で量子化します。  
静的量子化はキャリブレーションデータを用いて事前にテンソルのzero-point,scaleを計算する手法です。  


#### 動的量子化と静的量子化組み合わせ
組み合わせることもでき、速度と出力音質のバランスが良いです

### 推論
```
python separation_onnx.py   model.onnx   real_multiple_speakers_4mics.wav   outputs/tmp/   --n_channels 4 --sr 44100 --mic_radius .03231 
```
model.onnx: onnxのモデルです。変換して用意します  
real_multiple_speakers_4mics.wav: 4チャンネルの音声入力です(チャンネルの順はリスピーカー録音の順でOK.TODO: チャンネルの順番をドキュメント化. TODO: どのチャンネルの方向が0度か確認)__




### その他
separation.onnxはデフォルトでプロファイリングが有効になっていて、推論後にプロファイル結果のjsonのパスをprintします  
```
python analyze_profile.py onnxruntime_profile__2025-12-09_13-43-11.json
```
でどの演算がどれだけ時間がかかっているかを調べることができます
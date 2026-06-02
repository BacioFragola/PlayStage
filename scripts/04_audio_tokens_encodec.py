import torch
import librosa
from transformers import AutoProcessor, EncodecModel

# 1. 选择模型
model_name = "facebook/encodec_24khz"
processor = AutoProcessor.from_pretrained(model_name)
model = EncodecModel.from_pretrained(model_name)

# 【调试修改】：强行使用 CPU 排除显卡驱动问题
device = "cpu"
model = model.to(device)
print("Device forced to:", device)

# 2. 读取音频
audio_path = "data/VW_test_exchange_01.wav"
target_sr = 24000
audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

print("\nAudio waveform shape:", audio.shape)
print("Audio duration seconds:", len(audio) / target_sr)

print("\n[打卡 1] 开始进行音频预处理 (Processor)...")
inputs = processor(
    raw_audio=audio,
    sampling_rate=target_sr,
    return_tensors="pt"
)
inputs = {k: v.to(device) for k, v in inputs.items()}
print("[打卡 2] 音频预处理完成，数据已成功载入内存。")

print("\n[打卡 3] 开始进入神经网络编码 (model.encode), 带宽设置为 6.0...")
with torch.no_grad():
    encoded_outputs = model.encode(
        inputs["input_values"],
        inputs.get("padding_mask"),
        bandwidth=6.0
    )
print("[打卡 4] 神经网络编码成功完成！")

audio_codes = encoded_outputs.audio_codes
print("\nAudio codes shape:", audio_codes.shape)
print("Audio codes dtype:", audio_codes.dtype)
print("\nFirst 50 audio code IDs:")
print(audio_codes.flatten()[:50])
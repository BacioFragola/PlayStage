import os
import torch
import librosa
import matplotlib.pyplot as plt
from transformers import AutoProcessor, EncodecModel

# 1. 载入模型与音频 (强行使用 CPU 确保稳定)
model_name = "facebook/encodec_24khz"
processor = AutoProcessor.from_pretrained(model_name)
model = EncodecModel.from_pretrained(model_name).to("cpu")

audio_path = "data/VW_test_exchange_01.wav"
audio, sr = librosa.load(audio_path, sr=24000, mono=True)

# 2. 运行 EnCodec 提取 8 层 Tokens
inputs = processor(raw_audio=audio, sampling_rate=24000, return_tensors="pt")
with torch.no_grad():
    encoded_outputs = model.encode(inputs["input_values"], inputs.get("padding_mask"), bandwidth=6.0)

# 3. 提取矩阵并重塑形状
# 原始 shape: [1, 1, 8, 1125] -> 挤压掉前面的 1，变成 [8, 1125]
tokens_matrix = encoded_outputs.audio_codes.squeeze().cpu().numpy()

print("正在绘制热力图，当前矩阵形状为:", tokens_matrix.shape)

# 4. 开始绘制二维热力图
plt.figure(figsize=(15, 6))

# 使用 imshow 绘制矩阵，cmap='viridis' 会用蓝黄渐变色表示 Token ID 的大小
plt.imshow(tokens_matrix, aspect='auto', cmap='viridis', interpolation='nearest')

# 加上色彩刻度条（代表 Token ID 的数值，0~1024）
plt.colorbar(label='Audio Token ID')

# 整理坐标轴刻度
plt.title("Who's Afraid of Virginia Woolf? - EnCodec Audio Tokens Heatmap", fontsize=14)
plt.ylabel("Codebook Layers (1 to 8)", fontsize=12)
plt.xlabel("Time Steps (75 steps per second)", fontsize=12)

# 修改 Y 轴标签，使其对应 1~8 层
plt.yticks(range(8), [f"Layer {i+1}" for i in range(8)])

# 确保输出文件夹存在
os.makedirs("outputs", exist_ok=True)

# 保存图像
output_jpg = "outputs/stage_01_tokens.png"
plt.savefig(output_jpg, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n🎉 恭喜！画图成功！图像已保存在: {output_jpg}")
print("你可以直接去你的电脑文件夹里双击打开它查看！")
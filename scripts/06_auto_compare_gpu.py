import os
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from transformers import AutoProcessor, EncodecModel, pipeline

# 1. 动态检测 GPU (CUDA) 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 当前实验核心计算设备已强制切换至: 【{device.upper()}】")

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ========================================================
# 2. 切换至 Bark 精细控制的 Baseline 音频
# ========================================================
# 注释掉旧的 MMS 生成逻辑，不再重复生成
# tts_pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng", device=pipeline_device)
# tts_output = tts_pipe(text_prompt)

# 直接死锁我们用 07 脚本拼装好的高阶控制音频！
ai_wav_path = "data/VW_bark_controlled_full.wav" 
print(f"🎉 成功载入严格控制变量后的 Bark Baseline 音频！")


# ========================================================
# 3. GPU 加速提取离散 Token 矩阵与计算信息熵
# ========================================================
print("\n[打卡 3] 正在将 EnCodec 编码器载入 GPU 显存...")
model_name = "facebook/encodec_24khz"
processor = AutoProcessor.from_pretrained(model_name)
encodec_model = EncodecModel.from_pretrained(model_name).to(device)
encodec_model.eval() # 切换至评估模式，关闭不必要的梯度计算

def extract_metrics_gpu(audio_path):
    audio, _ = librosa.load(audio_path, sr=24000, mono=True)
    
    # 提取特征并将所有输入张量严格送入 GPU
    inputs = processor(raw_audio=audio, sampling_rate=24000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
    
    with torch.no_grad():
        encoded_outputs = encodec_model.encode(inputs["input_values"], inputs.get("padding_mask"), bandwidth=6.0)
    
    # 从 GPU 抓回内存转为 NumPy 矩阵
    matrix = encoded_outputs.audio_codes.squeeze().cpu().numpy() # [8, T]
    
    # 快速计算香农信息熵
    layer_entropies = []
    for layer in range(8):
        tokens = matrix[layer, :]
        _, counts = np.unique(tokens, return_counts=True)
        probs = counts / len(tokens)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        layer_entropies.append(entropy)
    return matrix, layer_entropies

real_wav_path = "data/VW_test_exchange_01.wav"
print("[打卡 4] GPU 正在并行并发处理双路音频特征...")
real_matrix, real_entropies = extract_metrics_gpu(real_wav_path)
ai_matrix, ai_entropies = extract_metrics_gpu(ai_wav_path)


# ========================================================
# 4. 打印论文 LaTeX 表格所需的硬核量化指标
# ========================================================
print("\n" + "="*20 + " 📊 论文表格量化数据 (GPU 计算完成) " + "="*20)
print(f"{'Codebook Layer':<18}{'Ground Truth (Real Drama)':<30}{'MMS-TTS Baseline (AI)':<30}")
print("-" * 78)
for i in range(8):
    print(f"Layer {i+1:<12}{real_entropies[i]:<30.4f}{ai_entropies[i]:<30.4f}")
print("-" * 78)
print(f"Mean Entropy    {np.mean(real_entropies):<30.4f}{np.mean(ai_entropies):<30.4f}")
print("=" * 78)


# ========================================================
# 5. 绘制双联并排热力图
# ========================================================
print("\n[打卡 5] 正在生成高分辨率论文插图...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

im1 = ax1.imshow(real_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
ax1.set_title("(a) Ground Truth (Real Actors - High-Tension/Hysteria Drama)", fontsize=11, loc='left', fontweight='bold')
ax1.set_ylabel("Codebook Layers")
ax1.set_yticks(range(8))
ax1.set_yticklabels([f"L{i+1}" for i in range(8)])

im2 = ax2.imshow(ai_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
ax2.set_title("(b) Baseline Synthesis (Standard TTS - Monotone/Flat)", fontsize=11, loc='left', fontweight='bold')
ax2.set_ylabel("Codebook Layers")
ax2.set_xlabel("Time Steps (13.33ms per step)")
ax2.set_yticks(range(8))
ax2.set_yticklabels([f"L{i+1}" for i in range(8)])

fig.colorbar(im2, ax=[ax1, ax2], label='Audio Token ID')
plt.suptitle("Acoustic Token Discrepancy Analysis", fontsize=14, fontweight='bold')

output_fig_path = "outputs/paper_comparison_heatmap_gpu.png"
plt.savefig(output_fig_path, bbox_inches='tight', dpi=300)
plt.close()

# 显式清理显存垃圾
if device == "cuda":
    torch.cuda.empty_cache()

print(f"🎉 显卡加速实验大成功！插图已保存至: {output_fig_path}")
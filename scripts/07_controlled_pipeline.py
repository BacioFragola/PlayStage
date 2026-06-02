import os
import json
import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, BarkModel

# ========================================================
# 1. 严格控制变量的戏剧文本矩阵
# ========================================================
experimental_manifest = [
    {
        "turn": 1,
        "character": "Martha",
        "speaker_preset": "v2/en_speaker_9",
        "text": "[sighs] Jesus... [gasp] H... Christ!",
        "output_name": "data/martha_turn_1_controlled.wav"
    },
    {
        "turn": 2,
        "character": "George",
        "speaker_preset": "v2/en_speaker_0",
        "text": "[whispering] Shhhh... For God’s sake, Martha, it’s after two o’clock.",
        "output_name": "data/george_turn_2_controlled.wav"
    },
    {
        "turn": 3,
        "character": "Martha",
        "speaker_preset": "v2/en_speaker_9",
        "text": "Oh, George! Well, I’m sorry, but... Oh, George!", 
        "output_name": "data/martha_turn_3_controlled.wav"
    },
    {
        "turn": 4,
        "character": "George",
        "speaker_preset": "v2/en_speaker_0",
        "text": "Well, I’m sorry, but... It’s late, you know? Late.",
        "output_name": "data/george_turn_4_controlled.wav"
    },
    {
        "turn": 5,
        "character": "Martha",
        "speaker_preset": "v2/en_speaker_9",
        "text": "What a cluck! What a cluck you are. What a dump.", 
        "output_name": "data/martha_turn_5_controlled.wav"
    }
]

os.makedirs("configs", exist_ok=True)
os.makedirs("data", exist_ok=True)
with open("configs/controlled_experiment_v1.json", "w", encoding="utf-8") as f:
    json.dump(experimental_manifest, f, indent=4, ensure_ascii=False)

# ========================================================
# 2. 启动免安装计算引擎
# ========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n📡 核心引擎就位！正在使用设备: 【{device.upper()}】")
print("⏳ 正在载入轻量级 Bark 处理器与模型...")

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to(device)
model.eval()

# 用一个列表来实时收集每一轮生成的音频片段，用于后续拼接
full_dialogue_signals = []
sr = 24000
# 设定对白之间 0.3 秒的自然换气静音块
silence_padding = np.zeros(int(sr * 0.3))

# ========================================================
# 3. 依照 Manifest 矩阵批量合成
# ========================================================
print("\n🎬 正在启动‘精细控制变量’对照组语音合成...")

for item in experimental_manifest:
    print(f"-> 正在合成 Turn {item['turn']} [{item['character']}]...")
    
    inputs = processor(item["text"], voice_preset=item["speaker_preset"], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        audio_output = model.generate(**inputs)
    
    audio_data = audio_output.squeeze().cpu().numpy()
    
    # 保存单句备份
    sf.write(item["output_name"], audio_data, sr)
    
    # 送入拼接池，并在句尾附赠一个换气停顿
    full_dialogue_signals.append(audio_data)
    full_dialogue_signals.append(silence_padding)

# ========================================================
# 4. 硬核追加：全自动时间轴级组装
# ========================================================
print("\n🔗 正在将分段对白无缝拼装为剧场版完整时间轴...")
# 矩阵级拼接
concatenated_audio = np.concatenate(full_dialogue_signals[:-1]) # 扔掉最后一个多余的静音块

combined_output_path = "data/VW_bark_controlled_full.wav"
sf.write(combined_output_path, concatenated_audio, sr)

print(f"🎉 [07 终极版] 拼装大成功！完整多角色对照音频已诞生: {combined_output_path}")
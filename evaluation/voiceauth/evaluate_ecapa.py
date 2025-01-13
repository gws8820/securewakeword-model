import os
import numpy as np
import torch
from speechbrain.inference import SpeakerRecognition
from pathlib import Path

enrollment_embedding_path = "../../model/voiceauth/SGW_ECAPA.npy"
test_dir = "../../dataset/Test"
model_source="speechbrain/spkrec-ecapa-voxceleb"

thresholds = [0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

spkrec = SpeakerRecognition.from_hparams(
    source=model_source,
    savedir="../../resources/pretrained_ecapa",
    run_opts={"device": device},
)
print(f"Loaded model from {model_source}")

enrolled_embedding = np.load(enrollment_embedding_path).squeeze()
print(f"Loaded enrolled embedding from {enrollment_embedding_path}")

print(f"{'Threshold':<10}{'FRR(%)':<10}{'FAR(%)':<10}")
print("-" * 30)

for threshold in thresholds:
    positive_cnt = negative_cnt = frr_cnt = far_cnt = 0
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                folder_name = os.path.basename(root)
                test_file = os.path.join(root, file)
                
                try:
                    test_emb = spkrec.encode_batch(spkrec.load_audio(test_file))
                    test_emb_np = test_emb.squeeze().detach().cpu().numpy()
                except Exception as e:
                    print(f"Error processing file {test_file}: {e}")
                    continue
                
                similarity = np.dot(enrolled_embedding, test_emb_np) / (
                    np.linalg.norm(enrolled_embedding) * np.linalg.norm(test_emb_np)
                )
                
                auth = similarity >= threshold
                
                if folder_name == "SGW":
                    positive_cnt += 1
                    if not auth:
                        frr_cnt += 1
                else:
                    negative_cnt += 1
                    if auth:
                        far_cnt += 1
    
    # FRR, FAR 계산
    frr_rate = (frr_cnt / positive_cnt * 100) if positive_cnt > 0 else 0.0
    far_rate = (far_cnt / negative_cnt * 100) if negative_cnt > 0 else 0.0
    
    # 결과 출력
    print(f"{threshold:<10.2f}{frr_rate:<10.2f}{far_rate:<10.2f}")
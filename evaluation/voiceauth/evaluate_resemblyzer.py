import os
import numpy as np
from resemblyzer import VoiceEncoder
import librosa
from pathlib import Path

encoder = VoiceEncoder("cpu")
embedding = np.load("../../model/voiceauth/SGW_resemblyzer.npy")
test_dir = "../../dataset/Test"

thresholds = [0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8]

print(f"{'Threshold':<16}{'FRR(%)':<11}{'FAR(%)':<11}")
print("-" * 38)
        
for threshold in thresholds:
    auth = True
    positive_cnt = negative_cnt = frr_cnt = far_cnt = 0
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                folder_name = os.path.basename(root)
                test_file = os.path.join(root, file)
                file_name = Path(test_file).stem
                
                try:
                    wav, sr = librosa.load(test_file, sr=16000)
                    test_emb = encoder.embed_utterance(wav)
                except Exception as e:
                    print(f"Error processing file {test_file}: {e}")
                    continue

                # 코사인 유사도 계산
                similarity = np.dot(embedding, test_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(test_emb)
                )
                
                auth = similarity >= threshold

                if(folder_name == "SGW"):
                    positive_cnt += 1
                    if(auth == False): frr_cnt += 1
                else:
                    negative_cnt += 1
                    if (auth == True): far_cnt += 1
    
    frr_rate = frr_cnt / positive_cnt * 100
    far_rate = far_cnt / negative_cnt * 100
    
    print(f"{threshold:<16.2}{frr_rate:<11.2f}{far_rate:<11.2f}")
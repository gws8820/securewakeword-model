import os
import numpy as np
import torch
from resemblyzer import VoiceEncoder
import librosa
from speechbrain.inference import SpeakerRecognition
from pathlib import Path

def get_simillarity(embedding, test_emb):
    similarity = np.dot(embedding, test_emb) / (
        np.linalg.norm(embedding) * np.linalg.norm(test_emb)
    )
    
    return similarity
    
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = VoiceEncoder(device)
print(f"Using device: {device}")

test_dir = "../../dataset/Test"

reseblyzer_thresholds = [0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8]
reseblyzer_embedding = np.load("../../model/voiceauth/SGW_resemblyzer.npy")

xvector_thresholds = [0.938, 0.939, 0.94, 0.941, 0.942]
xvector_embedding = np.load("../../model/voiceauth/SGW_xvector.npy").squeeze()
xvector_spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="../../resources/pretrained_xvector",
    run_opts={"device": device},
)

ecapa_thresholds = [0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42]
ecapa_embedding = np.load("../../model/voiceauth/SGW_ecapa.npy").squeeze()
ecapa_spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="../../resources/pretrained_ecapa",
    run_opts={"device": device},
)

print("=" * 50)
print("Model: Resemblyzer\n")
print(f"{'Threshold':<16}{'FRR(%)':<11}{'FAR(%)':<11}")
print("-" * 38)

for threshold in reseblyzer_thresholds:
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

                similarity = get_simillarity(reseblyzer_embedding, test_emb)
                auth = similarity >= threshold

                if(folder_name == "SGW"):
                    positive_cnt += 1
                    if(auth == False): frr_cnt += 1
                else:
                    negative_cnt += 1
                    if (auth == True): far_cnt += 1
    
    frr_rate = frr_cnt / positive_cnt * 100
    far_rate = far_cnt / negative_cnt * 100
    
    print(f"{threshold:<16.3}{frr_rate:<11.2f}{far_rate:<11.2f}")

print("=" * 50)
print("\nModel: x-vector\n")
print(f"{'Threshold':<16}{'FRR(%)':<11}{'FAR(%)':<11}")
print("-" * 38)

for threshold in xvector_thresholds:
    positive_cnt = negative_cnt = frr_cnt = far_cnt = 0
    
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                folder_name = os.path.basename(root)
                test_file = os.path.join(root, file)
                
                try:
                    test_emb = xvector_spkrec.encode_batch(xvector_spkrec.load_audio(test_file)).squeeze().detach().cpu().numpy()
                except Exception as e:
                    print(f"Error processing file {test_file}: {e}")
                    continue
                
                similarity = get_simillarity(xvector_embedding, test_emb)
                auth = similarity >= threshold
                
                if folder_name == "SGW":
                    positive_cnt += 1
                    if not auth:
                        frr_cnt += 1
                else:
                    negative_cnt += 1
                    if auth:
                        far_cnt += 1
    
    frr_rate = (frr_cnt / positive_cnt * 100) if positive_cnt > 0 else 0.0
    far_rate = (far_cnt / negative_cnt * 100) if negative_cnt > 0 else 0.0
    
    print(f"{threshold:<16.3}{frr_rate:<11.2f}{far_rate:<11.2f}")

print("=" * 50)
print("Model: ECAPA\n")
print(f"{'Threshold':<16}{'FRR(%)':<11}{'FAR(%)':<11}")
print("-" * 38)

for threshold in ecapa_thresholds:
    positive_cnt = negative_cnt = frr_cnt = far_cnt = 0

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                folder_name = os.path.basename(root)
                test_file = os.path.join(root, file)
                
                try:
                    test_emb = ecapa_spkrec.encode_batch(ecapa_spkrec.load_audio(test_file)).squeeze().detach().cpu().numpy()
                except Exception as e:
                    print(f"Error processing file {test_file}: {e}")
                    continue
                
                similarity = get_simillarity(ecapa_embedding, test_emb)
                auth = similarity >= threshold
                
                if folder_name == "SGW":
                    positive_cnt += 1
                    if not auth:
                        frr_cnt += 1
                else:
                    negative_cnt += 1
                    if auth:
                        far_cnt += 1

    frr_rate = (frr_cnt / positive_cnt * 100) if positive_cnt > 0 else 0.0
    far_rate = (far_cnt / negative_cnt * 100) if negative_cnt > 0 else 0.0

    print(f"{threshold:<16.3}{frr_rate:<11.2f}{far_rate:<11.2f}")
import os
import numpy as np
import torch
from resemblyzer import VoiceEncoder
import librosa
from speechbrain.inference import SpeakerRecognition

def get_similarity(embedding, test_emb):
    return np.dot(embedding, test_emb) / (np.linalg.norm(embedding) * np.linalg.norm(test_emb))

def generic_similarity_fn(file_path, config):
    if config["encoder"] is not None:
        wav, sr = librosa.load(file_path, sr=16000)
        test_emb = config["encoder"].embed_utterance(wav)
    else:
        audio = config["spkrec"].load_audio(file_path)
        test_emb = config["spkrec"].encode_batch(audio).squeeze().detach().cpu().numpy()
    return get_similarity(config["embedding"], test_emb)

def evaluate_model(test_dir, thresholds, similarity_fn, model_name):
    print("=" * 50)
    print(f"Model: {model_name}\n")
    print(f"{'Threshold':<16}{'FRR(%)':<11}{'FAR(%)':<11}")
    print("-" * 38)
    for threshold in thresholds:
        positive_cnt = negative_cnt = frr_cnt = far_cnt = 0
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(".wav"):
                    folder_name = os.path.basename(root)
                    file_path = os.path.join(root, file)
                    try:
                        similarity = similarity_fn(file_path)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        continue
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

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = VoiceEncoder(device)
print(f"Using device: {device}")

test_dir = "../dataset/Test"

model_configs = [
    {
        "name": "Resemblyzer",
        "thresholds": [0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8],
        "embedding": np.load("../model/voiceauth/SGW_resemblyzer.npy"),
        "encoder": encoder,
        "spkrec": None
    },
    {
        "name": "x-vector",
        "thresholds": [0.936, 0.937, 0.938, 0.939, 0.94, 0.941, 0.942, 0.943, 0.944, 0.945, 0.946],
        "embedding": np.load("../model/voiceauth/SGW_xvector.npy").squeeze(),
        "encoder": None,
        "spkrec": SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="../resources/pretrained_xvector",
            run_opts={"device": device},
        )
    },
    {
        "name": "ECAPA",
        "thresholds": [0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44],
        "embedding": np.load("../model/voiceauth/SGW_ecapa.npy").squeeze(),
        "encoder": None,
        "spkrec": SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="../resources/pretrained_ecapa",
            run_opts={"device": device},
        )
    }
]

for config in model_configs:
    similarity_fn = lambda fp, cfg=config: generic_similarity_fn(fp, cfg)
    evaluate_model(test_dir, config["thresholds"], similarity_fn, config["name"])
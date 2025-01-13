import os
import numpy as np
import torch
from resemblyzer import VoiceEncoder
import librosa

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

encoder = VoiceEncoder("cpu")
train_dir = "./dataset/Raw/SGW"
embeddings = []

for filename in os.listdir(train_dir):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(train_dir, filename)
        wav, sr = librosa.load(file_path, sr=16000)
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb)

enrolled_embedding = np.mean(embeddings, axis=0)
np.save("./model/voiceauth/SGW_resemblyzer.npy", enrolled_embedding)
print("Done. Saved enrolled embedding to ./model/voiceauth/SGW_resemblyzer.npy")
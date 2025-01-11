import os
import numpy as np
from resemblyzer import VoiceEncoder
import librosa

encoder = VoiceEncoder("cpu")

train_dir = "./Dataset/Raw/SGW"
embeddings = []

for filename in os.listdir(train_dir):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(train_dir, filename)
        wav, sr = librosa.load(file_path, sr=16000)
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb)

enrolled_embedding = np.mean(embeddings, axis=0)
np.save("./Model/SGW.npy", enrolled_embedding)
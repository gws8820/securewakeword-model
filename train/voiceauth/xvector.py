import os
import numpy as np
import torch
from speechbrain.pretrained import SpeakerRecognition

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="../../resources/pretrained_xvector",
    run_opts={"device": device},
)

train_dir = "../../dataset/Raw/SGW"
embeddings = []

for filename in os.listdir(train_dir):
    if filename.lower().endswith(".wav"):
        file_path = os.path.join(train_dir, filename)

        emb = spkrec.encode_batch(spkrec.load_audio(file_path))
        emb_np = emb.squeeze(0).detach().cpu().numpy()
        embeddings.append(emb_np)

enrolled_embedding = np.mean(embeddings, axis=0)
np.save("../../model/voiceauth/SGW_xvector.npy", enrolled_embedding)
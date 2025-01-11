from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os

input_directory = "../Dataset/Raw"
output_directory = "../Dataset/Raw_Trimmed"

for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.lower().endswith(".wav"):
            input_file_path = os.path.join(root, file)
            try:
                audio = AudioSegment.from_file(input_file_path, format="wav")
                silence_threshold = audio.dBFS - 25
                min_silence_len = 300
                nonsilent_ranges = detect_nonsilent(
                    audio,
                    min_silence_len=min_silence_len,
                    silence_thresh=silence_threshold
                )
                if not nonsilent_ranges:
                    print(f"모두 무음으로 판단됨: {input_file_path}")
                    continue
                trimmed_audio = AudioSegment.empty()
                for start_idx, end_idx in nonsilent_ranges:
                    trimmed_audio += audio[start_idx:end_idx]
                relative_path = os.path.relpath(root, input_directory)
                output_root = os.path.join(output_directory, relative_path)
                os.makedirs(output_root, exist_ok=True)
                output_file_path = os.path.join(output_root, file)
                trimmed_audio.export(output_file_path, format="wav")
                print(f"VAD 성공: {input_file_path} -> {output_file_path}")
            except Exception as e:
                print(f"VAD 실패: {input_file_path}: {e}")
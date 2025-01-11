from pydub import AudioSegment
from pydub.effects import normalize
import os

# 대상 디렉토리 설정
input_directory = "../Dataset/Raw"

# 디렉토리 내 모든 파일 탐색
for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.lower().endswith(".wav"):
            file_path = os.path.join(root, file)
            try:
                audio = AudioSegment.from_file(file_path)
                normalized_audio = normalize(audio)
                normalized_audio.export(file_path, format=os.path.splitext(file)[1][1:])
                print(f"표준화 성공: {file_path} -> {file_path}")
            except Exception as e:
                print(f"표준화 실패: {file_path}: {e}")
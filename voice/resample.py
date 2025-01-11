import os
from pydub import AudioSegment

# 대상 디렉토리 설정
input_directory = "../Dataset/Raw"

# 디렉토리 내 모든 파일 탐색
for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            
            try:
                audio = AudioSegment.from_file(file_path)
                audio_16k = audio.set_frame_rate(16000)
                audio_16k.export(file_path, format="wav")
                print(f"변환 완료: {file_path}")
            except Exception as e:
                print(f"변환 실패: {file_path}, {e}")
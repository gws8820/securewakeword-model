import os
import shutil
import random
from collections import defaultdict

# 대상 디렉토리 설정
input_directory = "../Dataset/Raw"
train_directory = "../Dataset/Train"
test_directory = "../Dataset/Test"

# Train/Test 결과 저장
results = defaultdict(lambda: {"train": 0, "test": 0})

# 디렉토리 생성 함수
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Train/Test 디렉토리 생성
ensure_directory_exists(train_directory)
ensure_directory_exists(test_directory)

# 디렉토리 내 모든 파일 탐색
for root, dirs, files in os.walk(input_directory):
    for file in files:
        if file.lower().endswith(".wav"):
            file_path = os.path.join(root, file)

            relative_folder = os.path.relpath(root, input_directory)
            
            # 랜덤으로 Train/Test 분리
            if random.random() < 0.7:
                shutil.copy(file_path, os.path.join(train_directory, file))
                results[relative_folder]["train"] += 1
            else:
                shutil.copy(file_path, os.path.join(test_directory, file))
                results[relative_folder]["test"] += 1

# 결과 출력
for folder, counts in results.items():
    print(f"Folder: {folder}")
    print(f"  Train: {counts['train']} files")
    print(f"  Test: {counts['test']} files")
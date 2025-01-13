from pathlib import Path
import wave

import openwakeword
from openwakeword.utils import bulk_predict
from openwakeword.metrics import get_false_positives

def get_duration(filepath):
    with wave.open(filepath, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / float(rate)

if __name__ == "__main__":
    # 디렉토리 설정
    model_dir = Path("../model/wakeword/onnx")
    positive_dir = Path("../dataset/Test")
    negative_dir = Path("../dataset/Conversation")
    
    # 모델 파일 탐색
    model_files = list(model_dir.glob("**/*.onnx"))
    if not model_files:
        raise FileNotFoundError("No model files found in the '../Model/onnx' directory.")
    
    # FRR 테스트 파일 로드
    positive_samples = list(positive_dir.glob("**/*.wav"))
    if not positive_samples:
        raise FileNotFoundError("No test files found in the '../Dataset/Test' directory.")
    
    # FAR 테스트 파일 로드 및 문자열로 변환
    negative_samples = list(negative_dir.glob("**/*.wav"))
    if not negative_samples:
        raise FileNotFoundError("No test files found in the '../Dataset/Conversation' directory.")
    negative_sample_paths = [str(p) for p in negative_samples]
    
    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    for threshold in thresholds:
        print("=" * 50)
        print(f"Threshold: {threshold}\n")
        # 표 헤더 출력 (모델명, FRR, FAR 열)
        print(f"{'Model':<16}{'FRR(%)':<11}{'FAR(/hr)':<11}")
        print("-" * 38)
        
        for model_file in model_files:
            model_name = model_file.stem
            # 모델 초기화
            owwModel = openwakeword.Model(
                wakeword_models=[str(model_file)],
                inference_framework="onnx"
            )
            
            # ---- FRR 평가 ----
            true_positives = 0
            false_negatives = 0
            
            # 각 긍정 샘플에 대해 예측 수행
            for positive_sample in positive_samples:
                predictions = owwModel.predict_clip(str(positive_sample))
                detected = any(list(pred.values())[0] >= threshold for pred in predictions)
                if detected:
                    true_positives += 1
                else:
                    false_negatives += 1
            
            total_samples = len(positive_samples)
            accuracy = (true_positives / total_samples) * 100 if total_samples > 0 else 0
            frr = 100.0 - accuracy
            
            # ---- FAR 평가 ----
            predictions_far = bulk_predict(
                file_paths = negative_sample_paths,
                wakeword_models=[str(model_file)],
                ncpu=10,
                inference_framework="onnx"
            )
            
            # 예측 결과 평탄화
            predictions_list = [i[model_name] for j in predictions_far.keys() for i in predictions_far[j]]
            
            # 허위 수락(FAR) 계산
            num_false_positives = get_false_positives(predictions_list, threshold=threshold, grouping_window=50)
            
            # 전체 대화 파일 길이 계산
            total_duration = sum(get_duration(str(fp)) for fp in negative_samples)
            total_audio_hours = total_duration / 3600 if total_duration > 0 else 0
            far_per_hour = num_false_positives / total_audio_hours if total_audio_hours > 0 else 0
            
            # 모델별 FRR과 FAR 출력 (테이블 행)
            print(f"{model_name:<16}{frr:<11.2f}{far_per_hour:<11.2f}")
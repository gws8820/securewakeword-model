import numpy as np
import sounddevice as sd
import soundfile as sf
import simpleaudio as sa
import os
import time
import threading
from pynput import keyboard
from pynput.keyboard import Key

paused = False
pause_event = threading.Event()
pause_event.set()

def on_press(key):
    global paused
    try:
        if key == Key.space:
            paused = not paused
            if paused:
                print("\n일시정지 되었습니다. 다시 시작하려면 스페이스바를 누르세요.")
                pause_event.clear()
            else:
                print("\n녹음을 재개합니다.")
                pause_event.set()
    except AttributeError:
        pass

def record_voice(duration, current, all, samplerate=44100, gain=1.0):
    print(f"\n({current}/{all}) 말하세요!")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    print("녹음 성공")
    amplified_recording = recording * gain

    amplified_recording = np.clip(amplified_recording, -1.0, 1.0)
    
    return amplified_recording

def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True 
    listener.start()

def main():
    global paused
    print("이 프로그램은 2초 간격으로 샘플 음성을 녹음합니다.\n일시정지하려면 대기 중 스페이스바를 누르세요.")
    
    name = input("\n이름을 입력하세요: ")
    folder_path = os.path.join(os.getcwd(), name)

    try:
        os.makedirs(folder_path, exist_ok=False)
        print(f"폴더 '{name}'이(가) 성공적으로 생성되었습니다.")
    except FileExistsError:
        print(f"폴더 '{name}'이(가) 이미 존재합니다. 기존 폴더를 사용합니다.")
        
    repeat = int(input("\n반복 횟수를 입력하세요 (숫자만): "))

    duration = 1  # 녹음 시간 (초)
    samplerate = 44100  # 샘플링 레이트
    gain = 4.0 # 증폭도

    # 키보드 리스너 시작
    start_keyboard_listener()

    print("\n곧 녹음이 시작됩니다..")
    for i in range(1, repeat + 1):
        pause_event.wait()

        # 2초 대기를 작은 간격으로 나누어 일시정지 체크
        for _ in range(20):
            if not pause_event.is_set():
                pause_event.wait()
            time.sleep(0.1)
        
        filename = f"{i:03}.wav"
        filepath = os.path.join(folder_path, filename)
        recording = record_voice(duration, i, repeat, samplerate, gain)

        sf.write(filepath, recording, samplerate)

    print("\n모든 녹음이 완료되었습니다.")

if __name__ == "__main__":
    main()
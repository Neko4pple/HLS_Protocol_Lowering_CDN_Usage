import subprocess
import os
import time
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# 입력 프레임 디렉토리 및 업스케일된 결과 디렉토리
input_frame_directory = "outputvideo/hls_1"  # 업스케일 작업에 사용되는 원본 프레임이 저장된 디렉토리
upscaled_output_directory = "outputvideo/hls1_out"  # 업스케일된 프레임이 저장되는 디렉토리

def count_frames(directory, extension=".jpg"):
    """
    주어진 디렉토리 내에서 특정 확장자를 가진 파일의 개수를 반환하는 함수.
    
    Parameters:
        directory (str): 파일을 세려는 디렉토리 경로.
        extension (str): 찾고자 하는 파일의 확장자 (기본값: ".jpg").
    
    Returns:
        int: 주어진 확장자를 가진 파일의 개수.
    """
    try:
        return len([name for name in os.listdir(directory) if name.endswith(extension)])
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return 0

initial_frame_count = count_frames(input_frame_directory, extension=".jpg")

@app.route('/notify', methods=['POST'])
def notify():
    # 원본 프레임 수 확인
    initial_frame_count = count_frames(input_frame_directory, extension=".jpg")
    print(f"Initial frame count in {input_frame_directory}: {initial_frame_count}")
    
    # 즉시 응답을 반환하여 요청 수신을 확인
    response_message = {"status": "received", "message": "Upscaling process started."}
    notify_response = jsonify(response_message)
    # 비동기적으로 모니터링 시작
    import threading
    threading.Thread(target=monitor_upscale).start()  
    return notify_response, 200
    # 별도의 스레드 또는 비동기 작업으로 ESRGAN 추론 및 업스케일링 진행


process = subprocess.Popen([
        "python", "inference_realesrgan.py",
        "--model_path", "experiments/pretrained_models/RealESRGAN_x4plus.pth",
        "--input", input_frame_directory
    ])
    
    # 백그라운드 작업으로 프레임 업스케일링 완료 여부 체크
def monitor_upscale():
    for _ in range(30000):  # 최대 30000초 동안 완료 여부 체크
        upscaled_frame_count = count_frames(upscaled_output_directory, extension=".jpg")
        print(f"Upscaled frame count in {upscaled_output_directory}: {upscaled_frame_count}")
        
        # 업스케일된 프레임 수가 초기 프레임 수와 동일할 경우 알림 전송
        if upscaled_frame_count >= initial_frame_count:
            try:
                complete_response = requests.post("http://video-processing-service:3000/complete")
                print("Completion notification sent successfully.")
                print("Response from video-processing-service:", complete_response.text)
            except requests.exceptions.RequestException as e:
                print("Error sending completion notification:", e)
            break

        time.sleep(2)  # 1초 대기 후 재시도
    else:
        print("Upscale output files are incomplete. Task may have failed.")

  
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3002)
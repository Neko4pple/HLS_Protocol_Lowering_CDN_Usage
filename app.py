import subprocess
import os
import threading
import time
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# 입력 프레임 디렉토리 및 업스케일된 결과 디렉토리
input_frame_directory = "outputvideo/hls_1"
upscaled_output_directory = "outputvideo/hls1_out"

def count_frames(directory, extension=".jpg"):
    try:
        return len([name for name in os.listdir(directory) if name.endswith(extension)])
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
        return 0

@app.route('/notify', methods=['POST'])
def notify():
    # 즉시 응답을 반환하여 요청 수신을 확인
    response_message = {"status": "received", "message": "Upscaling process started."}
    notify_response = jsonify(response_message)
    
    # 비동기적으로 ESRGAN 추론 및 업스케일링 진행
    threading.Thread(target=start_upscale_process).start()
    
    return notify_response, 200

def start_upscale_process():
    # subprocess로 ESRGAN 모델 실행
    process = subprocess.Popen([
        "python", "inference_realesrgan.py",
        "--model_path", "experiments/pretrained_models/RealESRGAN_x4plus.pth",
        "--input", input_frame_directory
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 터미널 출력을 실시간으로 읽어오는 부분
    for line in iter(process.stdout.readline, b''):
        decoded_line = line.decode('utf-8').strip()
        print(f"RealESRGAN output: {decoded_line}")

        if "finish" in decoded_line:  # 'finish' 문자열을 감지하면 완료 처리
            print("Upscaling finished.")
            # 업스케일 완료 후 알림을 전송
            try:
                complete_response = requests.post("http://video-processing-service:3000/complete")
                print("Completion notification sent successfully.")
                print("Response from video-processing-service:", complete_response.text)
            except requests.exceptions.RequestException as e:
                print("Error sending completion notification:", e)
            break

    process.stdout.close()
    process.wait()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3001)

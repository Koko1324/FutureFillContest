import cv2
import time
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime

# ================= 설정 =================
IMG_WIDTH, IMG_HEIGHT = 128, 128
MODEL_PATH = 'rain_classifier_model.h5'
IMG_SAVE_DIR = 'imageSave'
JSON_PATH = 'rain_status.json'
CAPTURE_INTERVAL = 10  # 초 단위, 이미지 캡처 주기

# ================= 모델 불러오기 =================
model = load_model(MODEL_PATH)

# ================= 이미지 예측 함수 =================
def predict_rain(img_path):
    try:
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        label = 'Rain' if prediction > 0.5 else 'No Rain'
        return label, float(prediction)
    except Exception as e:
        print("❌ 이미지 예측 실패:", e)
        return "Error", 0.0

# ================= JSON 저장 함수 =================
def save_result_to_json(label, confidence):
    result = {
        "time": datetime.now().isoformat(),
        "rain_detected": label,
        "confidence": confidence
    }
    with open(JSON_PATH, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"✅ 결과 저장 완료: {result}")

# ================= 메인 루프 =================
def main_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다.")
                continue

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            img_path = f"{IMG_SAVE_DIR}/{timestamp}.jpg"
            cv2.imwrite(img_path, frame)
            print(f"📸 이미지 저장: {img_path}")

            label, confidence = predict_rain(img_path)
            save_result_to_json(label, confidence)

            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("⏹️ 캡처 중단됨.")
    finally:
        cap.release()

if __name__ == "__main__":
    main_loop()

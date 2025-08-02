import tkinter as tk
import requests
from datetime import datetime
import modi_plus
import time
import threading
import logging
import math
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# ===================== 공통 상수 및 설정 =====================
API_KEY = '' # 여기에 API 키를 입력하세요.
URL = 'http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty'
GRADE_MAP = {
    '1': '좋음',
    '2': '보통',
    '3': '나쁨',
    '4': '매우나쁨',
    None: '정보없음'
}
SIDO_LIST = ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종',
             '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
SHAKE_THRESHOLD = 20.0
SAMPLE_INTERVAL = 0.2
MOTOR_SPEED_POSITIVE = 30
MOTOR_SPEED_NEGATIVE = -30
MOTOR_DURATION_OPEN_FULL = 2.5  # 창문을 활짝 여는 시간
MOTOR_DURATION_OPEN_HALF = 2.5 # 창문을 절반 여는 시간
MOTOR_DURATION_CLOSE = 2.5 # 창문을 닫는 시간

# AI 관련 설정
IMG_WIDTH, IMG_HEIGHT = 128, 128
MODEL_PATH = 'rain_classifier_model.h5'
IMG_SAVE_DIR = 'imageSave'
JSON_PATH = 'rain_status.json'
CAPTURE_INTERVAL = 10 # 초 단위, 이미지 캡처 주기

# ===================== 로깅 설정 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===================== 모디 초기화 =====================
try:
    bundle = modi_plus.MODIPlus()
    motor = bundle.motors[0]
    imu = bundle.imus[0]
    logger.info("✅ MODI+ 초기화 성공")
except Exception as e:
    logger.error(f"❌ MODI+ 초기화 실패: {e}")
    bundle, motor, imu = None, None, None

# ===================== 미세먼지 API 함수 =====================
def fetch_all_air_quality_data():
    all_results = []
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    for sido in SIDO_LIST:
        params = {
            'serviceKey': API_KEY,
            'returnType': 'json',
            'numOfRows': 1000,
            'pageNo': 1,
            'sidoName': sido,
            'ver': '1.0'
        }
        try:
            response = requests.get(URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            items = data.get('response', {}).get('body', {}).get('items', [])
            for item in items:
                all_results.append({
                    'Time': now,
                    'Station': item.get('stationName', ''),
                    'Sido': sido,
                    'PM10': item.get('pm10Value', '-'),
                    'Grade': GRADE_MAP.get(str(item.get('pm10Grade')), '정보없음')
                })
        except Exception as e:
            logger.warning(f"⚠️ API 에러({sido}): {e}")
            continue
    return all_results

# ===================== AI 관련 함수 =====================
try:
    rain_model = load_model(MODEL_PATH)
    logger.info("✅ AI 모델 로드 성공")
except Exception as e:
    logger.error(f"❌ AI 모델 로드 실패: {e}")
    rain_model = None

def predict_rain(img_path):
    if not rain_model:
        return "Error", 0.0
    try:
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = rain_model.predict(img_array, verbose=0)[0][0]
        label = 'Rain' if prediction > 0.5 else 'No Rain'
        return label, float(prediction)
    except Exception as e:
        logger.error(f"❌ 이미지 예측 실패: {e}")
        return "Error", 0.0

def save_result_to_json(label, confidence):
    result = {
        "time": datetime.now().isoformat(),
        "rain_detected": label,
        "confidence": confidence
    }
    with open(JSON_PATH, 'w') as f:
        json.dump(result, f, indent=4)
    logger.info(f"✅ AI 결과 저장 완료: {result}")

def webcam_and_ai_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("❌ 웹캠을 열 수 없습니다.")
        return

    try:
        os.makedirs(IMG_SAVE_DIR, exist_ok=True)
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("❌ 프레임을 읽을 수 없습니다.")
                time.sleep(CAPTURE_INTERVAL)
                continue

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            img_path = f"{IMG_SAVE_DIR}/{timestamp}.jpg"
            cv2.imwrite(img_path, frame)
            logger.info(f"📸 이미지 저장: {img_path}")
            
            label, confidence = predict_rain(img_path)
            save_result_to_json(label, confidence)
            
            time.sleep(CAPTURE_INTERVAL)

    except Exception as e:
        logger.error(f"❌ 웹캠 및 AI 처리 스레드 오류: {e}")
    finally:
        cap.release()
        logger.info("⏹️ 웹캠 캡처 중단됨.")

# ===================== GUI 앱 클래스 =====================
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("전국 미세먼지, 바람, 비 감지 프로그램")
        self.geometry("800x600")
        self.resizable(False, False)

        tk.Label(self, text="창문 자동 조절 프로그램", font=("Arial", 20)).place(relx=0.03, rely=0.03)
        tk.Label(self, text="미세먼지, 바람, 비를 고려해 창문을 자동 조절합니다.", font=("Arial", 13)).place(relx=0.03, rely=0.1)

        tk.Label(self, text="조회하고 싶은 지역명을 입력하세요", font=("Arial", 12)).place(relx=0.03, rely=0.2)
        self.entry_location = tk.Entry(self, width=30, font=("Arial", 12))
        self.entry_location.place(relx=0.03, rely=0.25)
        tk.Button(self, text="미세먼지 조회", font=("Arial", 12), command=self.check_air_quality).place(relx=0.25, rely=0.25)

        self.result_label = tk.Label(self, text="", font=("Arial", 12), fg="blue")
        self.result_label.place(relx=0.03, rely=0.32)

        self.wind_status_var = tk.StringVar(value="바람 감지: 대기 중...")
        self.wind_status_label = tk.Label(self, textvariable=self.wind_status_var, font=("Arial", 12), fg="purple")
        self.wind_status_label.place(relx=0.03, rely=0.40)
        
        self.rain_status_var = tk.StringVar(value="비 감지: 대기 중...")
        self.rain_status_label = tk.Label(self, textvariable=self.rain_status_var, font=("Arial", 12), fg="darkgreen")
        self.rain_status_label.place(relx=0.03, rely=0.48)

        self.saved_dust_grade = None
        self.all_air_data = []
        self.motor_lock = "idle"
        
        self.is_shaking = False
        
        # 스레드 시작
        threading.Thread(target=self.auto_update_air_quality, daemon=True).start()
        threading.Thread(target=webcam_and_ai_thread, daemon=True).start()
        
        self.prev_acc = imu.acceleration if imu else (0, 0, 0)
        self.after(int(SAMPLE_INTERVAL * 1000), self.update_imu_feedback)
        self.after(1000, self.update_rain_status)

        self.mainloop()

    def auto_update_air_quality(self):
        while True:
            try:
                self.all_air_data = fetch_all_air_quality_data()
                logger.info(f"✅ [갱신 성공] {len(self.all_air_data)}건의 데이터를 불러왔습니다.")
            except Exception as e:
                logger.error(f"❌ [자동갱신 실패] 미세먼지 데이터: {e}")
            time.sleep(300)

    def check_air_quality(self):
        station_name = self.entry_location.get().strip()
        if not station_name:
            self.result_label.config(text="❗ 지역명을 입력하세요.")
            return

        found = False
        for item in self.all_air_data:
            if item['Station'] == station_name:
                self.saved_dust_grade = item['Grade']
                self.result_label.config(
                    text=f"[{item['Sido']} - {item['Station']}] PM10: {item['PM10']}㎍/㎥ → {item['Grade']}"
                )
                self.control_motor_by_dust_grade() # 미세먼지 조회 시에만 실행
                found = True
                break

        if not found:
            self.result_label.config(text=f"❌ '{station_name}' 측정소 데이터를 찾을 수 없습니다.")
            self.saved_dust_grade = None

    def get_rain_status(self):
        try:
            with open(JSON_PATH, 'r') as f:
                data = json.load(f)
                return data.get('rain_detected', 'Unknown') == 'Rain'
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def update_rain_status(self):
        is_raining = self.get_rain_status()
        
        if is_raining:
            self.rain_status_var.set("비 감지: 비가 내리고 있습니다. (창문 닫힘)")
            self.rain_status_label.config(fg="red")
        else:
            self.rain_status_var.set("비 감지: 비가 내리지 않습니다.")
            self.rain_status_label.config(fg="darkgreen")

        # 비와 바람 감지는 계속해서 모터를 제어하도록 분리
        self.control_motor_by_wind_and_rain()
        self.after(1000, self.update_rain_status)

    def update_imu_feedback(self):
        if imu:
            try:
                curr_acc = imu.acceleration
                dx, dy, dz = (curr_acc[i] - self.prev_acc[i] for i in range(3))
                acc_diff = math.sqrt(dx * dx + dy * dy + dz * dz)
                
                self.is_shaking = acc_diff > SHAKE_THRESHOLD
                
                if self.is_shaking:
                    self.wind_status_var.set(f"[강풍 감지] 가속도 변화량: {acc_diff:.2f} (창문 닫힘)")
                    self.wind_status_label.config(fg="red")
                else:
                    self.wind_status_var.set(f"바람 감지: 정상 (가속도 변화량: {acc_diff:.2f})")
                    self.wind_status_label.config(fg="purple")

                self.prev_acc = curr_acc

                self.control_motor_by_wind_and_rain()
            except Exception as e:
                error_message = f"IMU 처리 오류: {e}"
                self.wind_status_var.set(f"바람 감지 오류: {e}")
                logger.error(error_message)

        self.after(int(SAMPLE_INTERVAL * 1000), self.update_imu_feedback)

    def control_motor_by_dust_grade(self):
        if not motor:
            logger.warning("MODI+ 모터 모듈을 찾을 수 없습니다. 모터 제어를 건너뜁니다.")
            return

        grade = self.saved_dust_grade
        
        if grade == '좋음':
            if self.motor_lock != "opening_full":
                print("🚦 미세먼지 좋음 → 창문 활짝 열기 시작...")
                self.motor_lock = "opening_full"
                motor.speed = MOTOR_SPEED_NEGATIVE
                threading.Thread(target=self.stop_motor_after_delay, args=(MOTOR_DURATION_OPEN_FULL,), daemon=True).start()
        elif grade == '보통':
            if self.motor_lock != "opening_half":
                print("🚦 미세먼지 보통 → 창문 조금 열기 시작...")
                self.motor_lock = "opening_half"
                motor.speed = MOTOR_SPEED_NEGATIVE
                threading.Thread(target=self.stop_motor_after_delay, args=(MOTOR_DURATION_OPEN_HALF,), daemon=True).start()
        elif grade in ['나쁨', '매우나쁨']:
            if self.motor_lock != "closing":
                print("🚦 미세먼지 나쁨/매우나쁨 → 창문 닫기 시작...")
                self.motor_lock = "closing"
                motor.speed = MOTOR_SPEED_POSITIVE
                threading.Thread(target=self.stop_motor_after_delay, args=(MOTOR_DURATION_CLOSE,), daemon=True).start()
        else:
            print("❔ 등급 정보 없음 또는 변경 없음 → 모터 동작 생략")

    def control_motor_by_wind_and_rain(self):
        if not motor:
            return
            
        should_close = False
        
        if self.is_shaking:
            should_close = True
        if self.get_rain_status():
            should_close = True

        # 비 또는 바람이 감지되면 창문을 닫음
        if should_close:
            if self.motor_lock != "closing":
                print("🚨 비/바람 감지 → 창문 닫기 시작...")
                self.motor_lock = "closing"
                motor.speed = MOTOR_SPEED_POSITIVE
                threading.Thread(target=self.stop_motor_after_delay, args=(MOTOR_DURATION_CLOSE,), daemon=True).start()

    def stop_motor_after_delay(self, delay):
        time.sleep(delay)
        motor.speed = 0
        self.motor_lock = "idle"
        print("🚦 모터 동작 종료")


if __name__ == "__main__":
    App()
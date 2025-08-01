import tkinter as tk
import requests
from datetime import datetime
import modi_plus
import time
import threading
import logging
import math

# ===================== 공통 상수 및 설정 =====================
API_KEY = ''
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
MOTOR_SPEED_POSITIVE = 90
MOTOR_SPEED_NEGATIVE = -90

# ===================== 로깅 설정 =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ===================== 모디 초기화 =====================
bundle = modi_plus.MODIPlus()
motor = bundle.motors[0]
imu = bundle.imus[0]

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
            print(f"⚠️ API 에러({sido}):", e)
            continue
    return all_results

# ===================== GUI 앱 클래스 =====================
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("전국 미세먼지 및 바람 감지 프로그램")
        self.geometry("800x600")
        self.resizable(False, False)

        tk.Label(self, text="창문 개폐 프로그램", font=("Arial", 20)).place(relx=0.03, rely=0.03)
        tk.Label(self, text="미세먼지 및 흔들림(바람)을 고려해 창문을 자동 조절합니다.", font=("Arial", 13)).place(relx=0.03, rely=0.1)

        tk.Label(self, text="조회하고 싶은 지역명을 입력하세요", font=("Arial", 12)).place(relx=0.03, rely=0.2)
        self.entry_location = tk.Entry(self, width=30, font=("Arial", 12))
        self.entry_location.place(relx=0.03, rely=0.25)
        tk.Button(self, text="미세먼지 조회", font=("Arial", 12), command=self.check_air_quality).place(relx=0.25, rely=0.25)

        self.result_label = tk.Label(self, text="", font=("Arial", 12), fg="blue")
        self.result_label.place(relx=0.03, rely=0.32)

        # New label for wind detection
        self.wind_status_var = tk.StringVar()
        self.wind_status_var.set("바람 감지: 대기 중...")
        self.wind_status_label = tk.Label(self, textvariable=self.wind_status_var, font=("Arial", 12), fg="purple")
        self.wind_status_label.place(relx=0.03, rely=0.40)

        self.saved_dust_grade = None
        self.all_air_data = []
        self.auto_update_csv()
        self.motor_lock = "idle"  # 모터 상태 플래그
        # Start IMU feedback in the main thread using after()
        self.prev_acc = imu.acceleration
        self.after(int(SAMPLE_INTERVAL * 1000), self.update_imu_feedback)

        self.mainloop()


    def auto_update_csv(self):
        try:
            self.all_air_data = fetch_all_air_quality_data()
            print(f"✅ [갱신 성공] {len(self.all_air_data)}건의 데이터를 불러왔습니다.")
        except Exception as e:
            print("❌ [자동갱신 실패]", e)
        self.after(300000, self.auto_update_csv)

    def check_air_quality(self):
        station_name = self.entry_location.get().strip()
        if not station_name:
            self.result_label.config(text="❗ 지역명을 입력하세요.")
            return

        for item in self.all_air_data:
            if item['Station'] == station_name:
                self.saved_dust_grade = item['Grade']
                self.result_label.config(
                    text=f"[{item['Sido']} - {item['Station']}] PM10: {item['PM10']}㎍/㎥ → {item['Grade']}"
                )
                threading.Thread(target=self.control_motor_by_dust_grade, daemon=True).start()
                return

        self.result_label.config(text=f"❌ '{station_name}' 측정소 데이터를 찾을 수 없습니다.")
        self.saved_dust_grade = None

    # 미세먼지에 의한 모터 제어 함수 수정
    def control_motor_by_dust_grade(self):
        grade = self.saved_dust_grade
        print("🚦 등급에 따른 모터 동작 시작...")

        self.motor_lock = "dust"  # 미세먼지 제어 시작

        try:
            if grade == '좋음':
                motor.speed = -70
                time.sleep(5)
            elif grade == '보통':
                motor.speed = -70
                time.sleep(2.5)
            elif grade in ['나쁨', '매우나쁨']:
                motor.speed = 70
                time.sleep(2.5)
            else:
                print("❔ 등급 정보 없음 → 모터 동작 생략")
                return
        finally:
            motor.speed = 0
            self.motor_lock = "idle"  # 제어 종료

    # IMU 업데이트 함수 수정
    def update_imu_feedback(self):
        try:
            curr_acc = imu.acceleration
            dx, dy, dz = (curr_acc[i] - self.prev_acc[i] for i in range(3))
            acc_diff = math.sqrt(dx * dx + dy * dy + dz * dz)

            if acc_diff > SHAKE_THRESHOLD:
                wind_message = f"[강풍 감지] 가속도 변화량: {acc_diff:.2f} (창문 닫힘)"
                if self.motor_lock == "idle":
                    self.motor_lock = "wind"
                    motor.speed = MOTOR_SPEED_POSITIVE
            else:
                wind_message = f"바람 감지: 정상 (가속도 변화량: {acc_diff:.2f})"
                if self.motor_lock == "wind":
                    motor.speed = 0
                    self.motor_lock = "idle"

            self.wind_status_var.set(wind_message)
            self.prev_acc = curr_acc

        except Exception as e:
            error_message = f"IMU 처리 오류: {e}"
            self.wind_status_var.set(f"바람 감지 오류: {e}")
            logger.error(error_message)
            if self.motor_lock == "wind":
                motor.speed = 0
                self.motor_lock = "idle"

        self.after(int(SAMPLE_INTERVAL * 1000), self.update_imu_feedback)

if __name__ == "__main__":
    App()
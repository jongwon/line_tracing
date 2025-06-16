import cv2, time, io, numpy as np
from YB_Pcb_Car import YB_Pcb_Car
from IPython.display import display
import ipywidgets as widgets
from PIL import Image
from scipy.signal import medfilt
from collections import deque

# ───────────────── 하드웨어 초기화 ─────────────────
car     = YB_Pcb_Car()
camera  = cv2.VideoCapture(0)
time.sleep(2)
car.Car_Stop()
car.Ctrl_Servo(1, 55)
car.Ctrl_Servo(2, 130)

# ───────────────── 필터 선택 ─────────────────
# 'kalman', 'moving_average', 'median' 중 선택
FILTER_TYPE = 'kalman'  # ← 여기서 필터 종류 변경!

print(f"🔧 사용중인 필터: {FILTER_TYPE}")

# ───────────────── 칼만 필터 클래스 ─────────────────
class KalmanFilter:
    def __init__(self, Q=0.01, R=3.0):
        self.Q, self.R = Q, R
        self.x = None       # 추정치
        self.P = 1.0        # 추정 오차 공분산
    
    def update(self, z):
        if self.x is None:          # 첫 샘플
            self.x = z
            return self.x
        # 예측
        P_pred = self.P + self.Q
        # 갱신
        K   = P_pred / (P_pred + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred
        return self.x

# ───────────────── 이동평균 필터 클래스 ─────────────────
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
    
    def update(self, z):
        self.buffer.append(z)
        return np.mean(self.buffer)

# ───────────────── 미디언 필터 클래스 ─────────────────
class MedianFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
    
    def update(self, z):
        self.buffer.append(z)
        return np.median(list(self.buffer))

# ───────────────── 필터 초기화 ─────────────────
if FILTER_TYPE == 'kalman':
    filter = KalmanFilter(Q=0.02, R=3.0)
elif FILTER_TYPE == 'moving_average':
    filter = MovingAverageFilter(window_size=5)
elif FILTER_TYPE == 'median':
    filter = MedianFilter(window_size=5)
else:
    raise ValueError(f"Unknown filter type: {FILTER_TYPE}")

# ───────────────── PID 제어기 클래스 ─────────────────
class PIDController:
    def __init__(self, kp=0.5, ki=0.05, kd=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        
    def update(self, error, dt=0.05):
        # 적분항 누적 (안티 와인드업 적용)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -50, 50)
        
        # 미분항 계산
        derivative = (error - self.prev_error) / dt
        
        # PID 출력
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        return output

# PID 초기화
pid = PIDController(kp=0.5, ki=0.05, kd=0.2)

# ───────────────── 영상 분석 함수 ─────────────────
def get_bias(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    
    h, w = bin_img.shape
    roi  = bin_img[int(h*0.6):, :]
    M    = cv2.moments(roi)
    
    if M['m00'] == 0:
        return None, bin_img
    
    cx   = int(M['m10']/M['m00'])
    # bias = cx - w//2
    # bias > 0: 라인이 화면 중앙보다 오른쪽에 있음 → 로봇이 왼쪽으로 벗어남 → 오른쪽 회전 필요
    # bias < 0: 라인이 화면 중앙보다 왼쪽에 있음 → 로봇이 오른쪽으로 벗어남 → 왼쪽 회전 필요
    return cx - w//2, bin_img

# ───────────────── 스트림 위젯 ─────────────────
stream = widgets.Image(format='jpeg', width=320, height=240)
display(stream)
jpg = lambda bgr: cv2.imencode('.jpg', bgr)[1].tobytes()

# ───────────────── 주행 파라미터 ─────────────────
# 필터별 최적 파라미터
if FILTER_TYPE == 'kalman':
    BASE_SPEED = 30
    GAIN = 0.6
    CLIP_CORR = 30
    MIN_PWM = 45
elif FILTER_TYPE == 'moving_average':
    BASE_SPEED = 28
    GAIN = 0.5
    CLIP_CORR = 25
    MIN_PWM = 45
elif FILTER_TYPE == 'median':
    BASE_SPEED = 28
    GAIN = 0.55
    CLIP_CORR = 28
    MIN_PWM = 45

# 급회전 파라미터 (공통)
HARD_TURN_SPEED = 60
THRESH_HARD = 75
THRESH_MEDIUM = 40

# 부드러운 가속/감속을 위한 변수
prev_left = 0
prev_right = 0
SMOOTH_FACTOR = 0.85

# ───────────────── 데이터 로깅용 변수 ─────────────────
log_data = {
    'time': [],
    'raw_bias': [],
    'filtered_bias': [],
    'left_motor': [],
    'right_motor': []
}
start_time = time.time()

# ───────────────── 메인 루프 ─────────────────
try:
    frame_count = 0
    while True:
        ok, frame = camera.read()
        if not ok:
            print("⚠️  카메라 프레임 수신 실패")
            time.sleep(0.5)
            continue
        
        frame = cv2.resize(frame, (320, 240))
        bias, bin_img = get_bias(frame)
        
        if bias is None:                      # 라인 분실
            # 부드럽게 정지
            prev_left = int(prev_left * 0.5)
            prev_right = int(prev_right * 0.5)
            if abs(prev_left) < 10 and abs(prev_right) < 10:
                car.Car_Stop()
                prev_left = prev_right = 0
            else:
                car.Control_Car(prev_left, prev_right)
            stream.value = jpg(bin_img)
            print("❌ 라인 미검출 → 부드럽게 정지")
            continue
        
        # 선택된 필터로 노이즈 제거
        bias_filtered = filter.update(bias)
        
        # PID 제어로 보정값 계산
        correction = pid.update(bias_filtered)
        correction = int(np.clip(correction, -CLIP_CORR, CLIP_CORR))
        
        # 단계별 회전 처리
        # bias > 0: 라인이 오른쪽에 있음 → 오른쪽으로 회전 필요
        # bias < 0: 라인이 왼쪽에 있음 → 왼쪽으로 회전 필요
        if abs(bias_filtered) > THRESH_HARD:
            # 급회전 - 양쪽 바퀴 반대 방향
            if bias_filtered > 0:  # 라인이 오른쪽 → 오른쪽 회전
                target_left = HARD_TURN_SPEED
                target_right = -HARD_TURN_SPEED // 3  # 오른쪽 역회전
            else:  # 라인이 왼쪽 → 왼쪽 회전
                target_left = -HARD_TURN_SPEED // 3  # 왼쪽 역회전
                target_right = HARD_TURN_SPEED
            print(f"⚠️  급회전  bias={bias_filtered:.1f}  L={target_left} R={target_right}")
            
        elif abs(bias_filtered) > THRESH_MEDIUM:
            # 중간 회전 - 양쪽 바퀴 모두 움직임
            turn_factor = 0.7
            base_turn = int(BASE_SPEED * 1.2)  # 회전시 기본 속도 약간 증가
            
            if bias_filtered > 0:  # 라인이 오른쪽 → 오른쪽 회전
                target_left = base_turn
                target_right = int(base_turn * turn_factor)
            else:  # 라인이 왼쪽 → 왼쪽 회전
                target_left = int(base_turn * turn_factor)
                target_right = base_turn
            print(f"➡️  중간회전  bias={bias_filtered:.1f}  L={target_left} R={target_right}")
            
        else:
            # 일반 주행 (PID 제어)
            # bias > 0: 오른쪽 회전 필요 → 왼쪽 빠르게, 오른쪽 느리게
            target_left = BASE_SPEED + correction
            target_right = BASE_SPEED - correction
            
            # 최소 PWM 보장
            if abs(target_left) < MIN_PWM:
                target_left = MIN_PWM if target_left >= 0 else -MIN_PWM
            if abs(target_right) < MIN_PWM:
                target_right = MIN_PWM if target_right >= 0 else -MIN_PWM
        
        # 부드러운 가속/감속 적용
        left = int(prev_left * (1 - SMOOTH_FACTOR) + target_left * SMOOTH_FACTOR)
        right = int(prev_right * (1 - SMOOTH_FACTOR) + target_right * SMOOTH_FACTOR)
        
        # 최소 PWM 보장 (멈춤 방지)
        if 0 < abs(left) < MIN_PWM:
            left = MIN_PWM if left > 0 else -MIN_PWM
        if 0 < abs(right) < MIN_PWM:
            right = MIN_PWM if right > 0 else -MIN_PWM
        
        # 모터 제어
        car.Control_Car(left, right)
        
        # 이전 값 저장
        prev_left = left
        prev_right = right
        
        # 화면 업데이트
        stream.value = jpg(bin_img)
        
        # 데이터 로깅 (나중에 분석용)
        current_time = time.time() - start_time
        log_data['time'].append(current_time)
        log_data['raw_bias'].append(bias)
        log_data['filtered_bias'].append(bias_filtered)
        log_data['left_motor'].append(left)
        log_data['right_motor'].append(right)
        
        # 디버깅 출력 - 실제 모터 명령값과 회전 방향 표시
        direction = "→" if bias_filtered > 0 else "←" if bias_filtered < 0 else "↑"
        print(f"[{FILTER_TYPE}] {direction} raw={bias:>4} filtered={bias_filtered:>6.1f}  L={left:>4} R={right:>4}")
        
        frame_count += 1
        time.sleep(0.02)  # 50Hz 제어 주기

except KeyboardInterrupt:
    print("\n🛑 프로그램 종료")
finally:
    car.Car_Stop()
    camera.release()
    cv2.destroyAllWindows()
    
    # 실험 결과 저장
    print(f"\n📊 실험 완료: {frame_count} 프레임 처리")
    print(f"📁 데이터 저장: {FILTER_TYPE}_log.npz")
    np.savez(f'{FILTER_TYPE}_log.npz', **log_data)
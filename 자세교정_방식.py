import cv2, time, io, numpy as np
from YB_Pcb_Car import YB_Pcb_Car
from IPython.display import display
import ipywidgets as widgets
from PIL import Image

# ──────────────────────── 하드웨어 초기화 ────────────────────────
car = YB_Pcb_Car()
camera = cv2.VideoCapture(0)
time.sleep(2)

car.Car_Stop()          # 안전을 위해 먼저 정지
car.Ctrl_Servo(1, 55)   # Pan  (필요하면 값 조정)
car.Ctrl_Servo(2, 140)  # Tilt

# ──────────────────────── 영상 분석 함수 ────────────────────────
def get_bias(frame):
    """라인의 중앙 좌표를 찾아 화면 중앙과의 오차(pixels) 반환"""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_img = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    
    h, w = bin_img.shape
    roi  = bin_img[int(h * 0.6):, :]          # 화면 하단 40 %만 사용
    M    = cv2.moments(roi)
    if M["m00"] == 0:                         # 흰색(라인) 면적이 없음
        return None, bin_img
    cx   = int(M["m10"] / M["m00"])           # 라인 중심 x좌표
    bias = cx - (w // 2)                      # +→오른쪽, −→왼쪽
    return bias, bin_img

# ──────────────────────── Jupyter 스트림 위젯 ────────────────────────
stream = widgets.Image(format="jpeg", width=320, height=240)
display(stream)

def jpg_bytes(bgr):
    return cv2.imencode(".jpg", bgr)[1].tobytes()

# ──────────────────────── 주행 파라미터 ────────────────────────
BASE_SPEED  = 35            # 직진 기본 PWM (25~45 사이에서 실험)
GAIN        = 0.8           # bias→보정 계수 (0.6~1.2 사이 튜닝)
CLIP_CORR   = 35            # correction 절댓값 최대치
MIN_PWM     = 55            # 모터가 겨우 움직이는 최소 PWM
HARD_TURN   = 70            # 강제 급회전 속도
THRESH_HARD = 70            # |bias|가 이 값보다 크면 급회전

# ──────────────────────── 메인 루프 ────────────────────────
try:
    while True:
        ok, frame = camera.read()
        if not ok:
            print("⚠️  카메라 프레임 수신 실패")
            time.sleep(0.5)
            continue

        frame = cv2.resize(frame, (320, 240))
        bias, bin_img = get_bias(frame)

        # ‣ 라인을 찾지 못했을 때
        if bias is None:
            car.Car_Stop()
            stream.value = jpg_bytes(bin_img)
            print("❌ 라인 미검출 → 정지")
            continue

        # ‣ 라인을 크게 벗어나면 급회전
        if abs(bias) > THRESH_HARD:
            if bias > 0:             # 라인이 오른쪽 → 차량을 오른쪽으로
                car.Control_Car(HARD_TURN, -HARD_TURN)
            else:                    # 라인이 왼쪽
                car.Control_Car(-HARD_TURN, HARD_TURN)
            stream.value = jpg_bytes(bin_img)
            print(f"⚠️  급회전  bias={bias}")
            continue

        # ‣ 일반 구간: 비례 보정 + 클리핑
        corr = int(np.clip(bias * GAIN, -CLIP_CORR, CLIP_CORR))

        left  = BASE_SPEED - corr
        right = BASE_SPEED + corr

        # 최소 PWM 보장 (정지·역주행 방지)
        if abs(left)  < MIN_PWM: left  = MIN_PWM if left  >= 0 else -MIN_PWM
        if abs(right) < MIN_PWM: right = MIN_PWM if right >= 0 else -MIN_PWM

        car.Control_Car(left, right)
        stream.value = jpg_bytes(bin_img)
#         print(f"bias={bias:>4}  L={left:>4}  R={right:>4}")

finally:
    car.Car_Stop()
    camera.release()
    cv2.destroyAllWindows()


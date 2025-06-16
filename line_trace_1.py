import cv2
import numpy as np
import time
from YB_Pcb_Car import YB_Pcb_Car
from PIL import Image
import io


car = YB_Pcb_Car()
camera = cv2.VideoCapture(0)
time.sleep(2)


car.Car_Stop()
car.Ctrl_Servo(1, 55)  # Pan 초기화
car.Ctrl_Servo(2, 150) # Tilt 초기화

# 추가 코드
def get_bias(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 자동 이진화 (adaptive or Otsu)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape
    roi = binary[int(h * 0.6):, :]

    moments = cv2.moments(roi)
    if moments["m00"] == 0:
        return None, binary  # 라인을 못 찾음
    cx = int(moments["m10"] / moments["m00"])
    bias = cx - (w // 2)
    return bias, binary



try:
    import ipywidgets as widgets
    from IPython.display import display
    import cv2, time

    stream_widget = widgets.Image(format='jpeg', width=320, height=240)
    display(stream_widget)

    def bgr_to_jpeg(bgr):
        return cv2.imencode('.jpg', bgr)[1].tobytes()

    base_speed = 10
    gain = 0.4
    min_pwm = 55

    while True:
        ret, frame = camera.read()
        if not ret:
            print("⚠️ 카메라 안 보임")
            time.sleep(1)
            continue

        frame = cv2.resize(frame, (320, 240))
        bias, binary_img = get_bias(frame)

        if bias is None:
            car.Car_Stop()
            print("⚠️ 라인 없음 → 정지")
            stream_widget.value = bgr_to_jpeg(binary_img)
            continue

        correction = int(bias * gain)
        left_speed = base_speed - correction
        right_speed = base_speed + correction

        for v in ("left_speed", "right_speed"):
            val = locals()[v]
            if abs(val) < min_pwm:
                locals()[v] = min_pwm if val >= 0 else -min_pwm

        car.Control_Car(left_speed, right_speed)
        print(f"bias={bias:4d}  L={left_speed:4d}  R={right_speed:4d}")

        # 스트림 화면 덮어쓰기
        stream_widget.value = bgr_to_jpeg(binary_img)

finally:
    car.Car_Stop()
    camera.release()
    cv2.destroyAllWindows()



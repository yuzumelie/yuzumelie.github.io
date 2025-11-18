import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from PIL import ImageFont, ImageDraw, Image
import os

# ==========================
# 0. 경로 설정 (여기만 수정!)
# ==========================

AUDIO_FOLDER = r"/Users/millie/Documents/audio"  # mp3가 있는 폴더 절대 경로로 수정
FONT_PATH = r"/System/Library/Fonts/AppleSDGothicNeo.ttc"  # macOS 기본 한글 폰트 경로

# ==========================
# 1. 오디오 초기화
# ==========================

pygame.mixer.init()

guide_states = ["move_back", "come_in"]
sounds = {}

for state in guide_states:
    path = os.path.join(AUDIO_FOLDER, f"{state}.mp3")
    if os.path.exists(path):
        sounds[state] = pygame.mixer.Sound(path)
        print(f"[Loaded] {path}")
    else:
        print(f"[Missing] {path}")

last_audio_play_time = 0
current_state = None

def play_guide(state, cooldown=2.0):
    """
    음성 재생 규칙:
    - perfect는 재생 안 함
    - come_in, move_back 중 하나가 재생되면
      그 후 2초 동안 다른 음성은 재생되지 않음
    - 단, 같은 음성은 cooldown 안에서만 차단
    """

    global last_audio_play_time, current_state

    if state == "perfect":
        current_state = state
        return

    now = time.time()
    time_since_last = now - last_audio_play_time

    # -----------------------------
    # 1) 이전 음성과 다른 음성인데,
    #    마지막 재생 이후 2초가 안 지났으면 차단
    # -----------------------------
    if current_state is not None and current_state != state:
        if time_since_last < 2.0:
            return  # 다른 음성을 차단

    # -----------------------------
    # 2) 동일 음성 재생 쿨다운
    # -----------------------------
    if current_state == state and time_since_last < cooldown:
        return

    # -----------------------------
    # 3) 재생
    # -----------------------------
    if state in sounds:
        sounds[state].play()
        print("[AUDIO]", state)

    current_state = state
    last_audio_play_time = now





# ==========================
# 2. 텍스트 출력
# ==========================

try:
    font = ImageFont.truetype(FONT_PATH, 24)
    USE_PIL_TEXT = True
except:
    USE_PIL_TEXT = False
    font = None

def draw_text(frame, text, is_good=False):
    h, w, _ = frame.shape
    color = (0, 255, 0) if is_good else (255, 255, 255)

    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)

    if USE_PIL_TEXT:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.text((10, 5), text, font=font, fill=color)
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# ==========================
# 3. 얼굴 분석 로직
# ==========================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
def facial_features_visible(face, h, w):
    # 주요 이목구비 landmark
    nose_id = 1
    left_eye_id = 33
    right_eye_id = 263
    mouth_id = 13

    key_ids = [nose_id, left_eye_id, right_eye_id, mouth_id]

    for idx in key_ids:
        lm = face.landmark[idx]

        # landmark가 프레임 밖이면 False
        if lm.x < 0 or lm.x > 1 or lm.y < 0 or lm.y > 1:
            return False
    
    return True
def analyze_faces(multi_face_landmarks, w, h):

    if not multi_face_landmarks:
        return "화면 안으로 들어오세요", "come_in", False

    face = multi_face_landmarks[0]

    xs = [lm.x for lm in face.landmark]
    ys = [lm.y for lm in face.landmark]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    bw = max_x - min_x
    bh = max_y - min_y

    # 눈 위치
    eye_ids = [33, 133, 362, 263]
    eye_ys = [face.landmark[i].y for i in eye_ids]
    avg_eye_y = sum(eye_ys) / len(eye_ys)

    # -------------------------------
    # 1) 🔥 “진짜 가까움” (visible_ratio 무시)
    # -------------------------------
    # bw, bh는 0~1 범위에서 얼굴이 차지하는 비율.
    # 0.70 이상이면 화면에 거의 얼굴만 꽉 찬 상태.
    if bw > 0.70 or bh > 0.70:
        return "조금 뒤로 물러나세요", "move_back", False

    # -------------------------------
    # 2) 얼굴 보이는 비율 계산
    # -------------------------------
    vis_x0 = np.clip(min_x, 0, 1)
    vis_x1 = np.clip(max_x, 0, 1)
    vis_y0 = np.clip(min_y, 0, 1)
    vis_y1 = np.clip(max_y, 0, 1)

    vis_w = (vis_x1 - vis_x0) / bw if bw > 0 else 0
    vis_h = (vis_y1 - vis_y0) / bh if bh > 0 else 0
    visible_ratio = min(vis_w, vis_h)

    # -------------------------------
    # 3) 절반 이상 프레임 밖 → come_in
    # -------------------------------
    if visible_ratio < 0.5:
        return "화면 안으로 들어오세요", "come_in", False

    # -------------------------------
    # 4) 눈 위치가 너무 위 → come_in
    # -------------------------------
    if avg_eye_y < 0.15:
        return "화면 안으로 들어오세요", "come_in", False

    # -------------------------------
    # 5) 정상
    # -------------------------------
    return "완벽합니다!", "perfect", True



# ==========================
# 4. 메인 루프 (히스테리시스 안정화 적용)
# ==========================

# 🔥 상태 안정화용 변수
stable_msg = None
candidate_msg = None
candidate_count = 0
REQUIRED_FRAMES = 12   # 12프레임 연속 유지 시 상태 확정 (30fps≈0.4초)

def main():
    global stable_msg, candidate_msg, candidate_count

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    print("😊 웹캠 시작! 종료 키: 'q'")

    start_time = time.time()
    WARMUP_SECONDS = 1.5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # 얼굴 분석
            if results.multi_face_landmarks:
                msg, guide_state, good = analyze_faces(results.multi_face_landmarks, w, h)
            else:
                msg, guide_state, good = "화면 안으로 들어오세요", "come_in", False

            # ==================================================
            # 🔥 상태 히스테리시스 (프레임 기반 안정화)
            # ==================================================
            if candidate_msg != msg:
                candidate_msg = msg
                candidate_count = 1
            else:
                candidate_count += 1

            # 일정 프레임 유지해야 상태 확정
            if candidate_count >= REQUIRED_FRAMES:
                stable_msg = candidate_msg

            # 자막은 안정화된 stable_msg로 표시
            final_msg = stable_msg if stable_msg is not None else msg

            # 자막 출력
            draw_text(frame, final_msg, is_good=good)

            # 음성 재생 (WARMUP 후)
            if time.time() - start_time > WARMUP_SECONDS:
                if guide_state != "perfect":
                    play_guide(guide_state)

            # 화면 표시
            cv2.imshow("Face Guide", frame)

            # 종료 키 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()



if __name__ == "__main__":
    main()

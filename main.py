import os
import random
import cv2
import time
import numpy as np
import pygame
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image

# =========================
#  論文展示版設定（可調）
# =========================
WINDOW_TITLE = "Emotion-Based Music Demo (Thesis)"
FONT_PATH = "NotoSansTC-VariableFont_wght.ttf"

# 顯示/計數策略（展示用）
EMOTIONS = ["happy", "sad", "angry"]
EMOTION_TRANSLATION = {"happy": "快樂", "sad": "悲傷", "angry": "生氣"}

EMOTION_THRESHOLD = 10          # 連續/累積達到幾次才觸發
RESET_INTERVAL_SEC = 180        # 超過多久沒有觸發就重置（秒）
HOLD_DISPLAY_MS = 1500          # 觸發後畫面停留多久（展示用，毫秒）

# ======== 商業保護：音樂策略 ========
# 展示版不公開你的音樂資料夾結構、檔名、挑選策略
# 你可以選擇：
# 1) DEMO_PLAY_MODE = "none"：不播放，只顯示提示（最安全）
# 2) DEMO_PLAY_MODE = "beep"：播放固定「提示音」示意（仍安全）
DEMO_PLAY_MODE = "none"  # "none" or "beep"

# 若用 beep 模式，可放一個固定 demo 音效檔，不含商業音樂庫
DEMO_BEEP_PATH = "demo_beep.wav"  # 你可自行準備一個很短的提示音（非商用音樂）


def safe_demo_audio_trigger(dominant_emotion: str):
    """
    論文展示版：只做示意，不暴露商業音樂資料夾與檔案。
    """
    if DEMO_PLAY_MODE == "none":
        print(f"[DEMO] 情緒觸發：{dominant_emotion}（展示版不播放音樂）")
        return

    if DEMO_PLAY_MODE == "beep":
        try:
            if not os.path.exists(DEMO_BEEP_PATH):
                print(f"[DEMO] 找不到 demo 音效檔：{DEMO_BEEP_PATH}（略過播放）")
                return

            pygame.mixer.music.load(DEMO_BEEP_PATH)
            pygame.mixer.music.play()
            print(f"[DEMO] 已播放提示音（情緒：{dominant_emotion}）")
        except Exception as e:
            print(f"[DEMO] 播放提示音失敗：{e}")


def put_chinese_text(img, text, position, font_path, font_size, color):
    """
    用 PIL 在 OpenCV 畫面上畫中文
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img


def calc_percentage(counts: dict, emotion: str) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return (counts.get(emotion, 0) / total) * 100.0


def main():
    # 初始化 pygame
    pygame.mixer.init()

    # 初始化攝影機
    cap = cv2.VideoCapture(0)

    # 載入臉部偵測分類器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    emotion_counts = {k: 0 for k in EMOTIONS}
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        triggered = False

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            # DeepFace 情緒分析（展示版仍保留）
            results = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)

            # results 通常是 list，取第一個
            result = results[0] if isinstance(results, list) and results else results
            emotion = result.get("dominant_emotion", None)

            if emotion not in emotion_counts:
                continue

            # 累積計數
            emotion_counts[emotion] += 1

            # 尚未達到門檻：只畫框，不觸發播放示意
            cv2.rectangle(frame, (x, y), (x + w, y + h), (160, 32, 240), 2)
            p = calc_percentage(emotion_counts, emotion)

            # 顯示「目前判定」而不是商用策略（避免泄漏你真正判斷/挑歌規則）
            info_text = f"目前情緒: {EMOTION_TRANSLATION[emotion]}  ({p:.1f}%)"
            frame = put_chinese_text(frame, info_text, (x, max(0, y - 30)),
                                     FONT_PATH, 24, (160, 32, 240))

            # 達到門檻：展示觸發
            if emotion_counts[emotion] >= EMOTION_THRESHOLD:
                triggered = True
                final_p = calc_percentage(emotion_counts, emotion)

                trigger_text = f"觸發情緒: {EMOTION_TRANSLATION[emotion]}  ({final_p:.1f}%)"
                frame = put_chinese_text(frame, trigger_text, (x, max(0, y - 60)),
                                         FONT_PATH, 28, (255, 255, 255))

                cv2.imshow(WINDOW_TITLE, frame)
                cv2.waitKey(HOLD_DISPLAY_MS)

                # 商業保護：只做示意，不播放你的音樂庫
                safe_demo_audio_trigger(emotion)

                # 重置
                emotion_counts = {k: 0 for k in EMOTIONS}
                start_time = time.time()
                break  # 一次只處理一張臉觸發，便於展示

        # 超過重置時間
        if time.time() - start_time >= RESET_INTERVAL_SEC:
            print("[DEMO] 超過重置時間，重新計數")
            emotion_counts = {k: 0 for k in EMOTIONS}
            start_time = time.time()

        # 顯示影像
        cv2.imshow(WINDOW_TITLE, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

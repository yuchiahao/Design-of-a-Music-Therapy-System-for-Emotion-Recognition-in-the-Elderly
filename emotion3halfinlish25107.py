import os
import random
import cv2
import time
import numpy as np
import pygame
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image

# 初始化 pygame
pygame.mixer.init()

# 初始化攝影機
cap = cv2.VideoCapture(0)

# 載入臉部偵測分類器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 建立一個字典來將英文的情緒名稱映射到中文
emotion_translation = {
    'happy': '快樂',
    'sad': '悲傷',
    'angry': '生氣'
}

# 初始化情緒計數和計時器
emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0}
emotion_threshold = 10  # 設定情緒計數閾值
start_time = time.time()
reset_interval = 180  # 重置間隔為180秒（3分鐘）

def random_music(dominant_emotion):
    try:
        # 根據主要情緒選擇對應的音樂資料夾
        emotion_folder_map = {
            'happy': 'C:/Ai_project/vm_project/vm_project2/music_folder/happy',  # 替換為實際的資料夾路徑
            'sad': 'C:/Ai_project/vm_project/vm_project2/music_folder/sad',
            'angry': 'C:/Ai_project/vm_project/vm_project2/music_folder/angry'
        }

        # 獲取對應資料夾的路徑
        music_folder = emotion_folder_map.get(dominant_emotion)
        if not music_folder:
            print(f"未找到對應的音樂資料夾: {dominant_emotion}")
            return

        # 獲取資料夾中的所有音樂檔案
        music_files = [f for f in os.listdir(music_folder) if f.endswith('.mp3')]
        if not music_files:
            print(f"資料夾中沒有音樂檔案: {music_folder}")
            return

        # 隨機選擇一個音樂檔案
        selected_music = random.choice(music_files)
        music_path = os.path.join(music_folder, selected_music)

        # 播放音樂
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"播放音樂時發生錯誤: {e}")

def put_chinese_text(img, text, position, font_path, font_size, color):
    # 將 OpenCV 圖像轉換為 PIL 圖像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    # 將 PIL 圖像轉換回 OpenCV 圖像
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

# 主要程式邏輯
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 將影像轉換為灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 偵測臉部
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 提取臉部的 ROI (感興趣區域)
        face_roi = frame[y:y + h, x:x + w]

        # 在臉部ROI上進行情緒分析
        results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # 打印模型的完整輸出
        print("情緒分析結果:", results)

        # 確定主導的情緒
        result = results[0]  # 提取列表中的第一個元素
        emotion = result['dominant_emotion']

        # 只處理 happy, sad 和 angry 情緒
        if emotion in emotion_counts:
            # 更新情緒計數
            emotion_counts[emotion] += 1

            # 檢查是否有情緒累積次數達到閾值
            if emotion_counts[emotion] >= emotion_threshold:
                # 計算並顯示每種情緒的百分比
                percentage = (emotion_counts[emotion] / sum(emotion_counts.values())) * 100
                print(f"主要情緒: {emotion} ({percentage:.2f}%)")
                # 在臉部周圍畫矩形並標記預測的情緒和百分比
                cv2.rectangle(frame, (x, y), (x + w, y + h), (160, 32, 240), 2)
                # 使用自定義字體顯示中文字符
                text = f"{emotion_translation[emotion]}: {percentage:.2f}%"
                frame = put_chinese_text(frame, text, (x, y - 30), "C:/Ai_project/vm_project/vm_project2/NotoSansTC-VariableFont_wght.ttf", 24, (160, 32, 240))
                # 顯示影像並停留五秒
                cv2.imshow('Yu chia hao', frame)
                cv2.waitKey(5000)  # 停留五秒
                # 播放對應情緒的音樂
                random_music(emotion)
                # 重置計數和計時器
                emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0}
                start_time = time.time()
                break

    # 檢查是否超過重置間隔
    total_elapsed_time = time.time() - start_time
    if total_elapsed_time >= reset_interval:
        print("超過三分鐘，重新計時")
        # 重置計數和計時器
        emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0}
        start_time = time.time()

    # 顯示影像
    cv2.imshow('Yu chia hao', frame)

    # 每五秒偵測一次
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機資源並關閉所有 OpenCV 視窗
cap.release()
cv2.destroyAllWindows()
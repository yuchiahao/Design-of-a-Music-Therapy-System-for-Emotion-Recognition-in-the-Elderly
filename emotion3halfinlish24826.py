import cv2
import time
from deepface import DeepFace

# 初始化情緒計數和計時器
emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0}
start_time = time.time()
reset_interval = 180  # 重置間隔為180秒（3分鐘）

# 初始化攝影機
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 將影像轉換為 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 偵測臉部
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 提取臉部的 ROI (感興趣區域)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # 在臉部ROI上進行情緒分析
        results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # 打印模型的完整輸出
        print("情緒分析結果:", results)

        # 確定主導的情緒
        result = results[0]  # 提取列表中的第一個元素
        emotion = result['dominant_emotion']

        # 更新情緒計數
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1

        # 檢查是否有情緒累積次數達到十次
        total_emotions = sum(emotion_counts.values())
        if total_emotions >= 10:
            # 計算並顯示每種情緒的百分比
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            percentage = (emotion_counts[dominant_emotion] / total_emotions) * 100
            print(f"主要情緒: {dominant_emotion} ({percentage:.2f}%)")
            # 在臉部周圍畫矩形並標記預測的情緒和百分比
            cv2.rectangle(frame, (x, y), (x + w, y + h), (160, 32, 240), 2)
            cv2.putText(frame, f"{dominant_emotion}: {percentage:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 32, 240), 2)
            # 顯示影像並停留五秒
            cv2.imshow('Emotion Detection', frame)
            cv2.waitKey(5000)  # 停留五秒
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
    cv2.imshow('Emotion Detection', frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機資源並關閉所有 OpenCV 視窗
cap.release()
cv2.destroyAllWindows()
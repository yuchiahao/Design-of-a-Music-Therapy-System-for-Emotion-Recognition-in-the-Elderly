import cv2
from deepface import DeepFace
import time

# 載入臉部級聯分類器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#  開始擷取影片
cap = cv2.VideoCapture(0)

# 創建一個字典來存儲每種情緒的出現次數
emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0, 'fear': 0, 'disgust': 0, 'surprise': 0}

# 記錄開始時間
start_time = time.time()





#  開始擷取影片
cap = cv2.VideoCapture(0)
# 初始化情緒計數字典
emotion_counts = {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 0, 'fear': 0, 'disgust': 0, 'surprise': 0}

while True:
    #"逐幀擷取
    ret, frame = cap.read()

    #  將畫面轉換為灰階
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 將灰階畫面轉換為 RGB 格式
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # 在畫面中偵測臉部
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 提取臉部的 ROI (感興趣區域)
        face_roi = rgb_frame[y:y + h, x:x + w]

        
        # 在臉部ROI上進行情緒分析
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # 確定主導的情緒
        emotion = result[0]['dominant_emotion']

            # 更新情緒計數
    if emotion in emotion_counts:
        emotion_counts[emotion] += 1

        # 在臉部周圍畫矩形並標記預測的情緒
        cv2.rectangle(frame, (x, y), (x + w, y + h), (160, 32, 240), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 32, 240), 2)

    # 顯示結果框架
    cv2.imshow('chia hao', frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




    # 逐幀擷取
    ret, frame = cap.read()

    #  將畫面轉換為灰階
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 將灰階畫面轉換為 RGB 格式
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # 在畫面中偵測臉部
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # 提取臉部的 ROI (感興趣區域)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # 對每一個偵測到的臉部進行情緒分析
        try:
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            if 'dominant_emotion' in analysis:
                dominant_emotion = analysis['dominant_emotion']
                # 增加該情緒的計數
                if dominant_emotion in emotion_counts:
                    emotion_counts[dominant_emotion] += 1

                # 在鏡頭畫面上顯示臉部辨識框和情緒
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        except Exception as e:
            print(f"分析失敗: {str(e)}")

   
    cv2.imshow('chia hao', frame)

 
    current_time = time.time()
    if current_time - start_time > 30:
        break


total_emotions = sum(emotion_counts.values())
for emotion, count in emotion_counts.items():
    percentage = (count / total_emotions) * 100 if total_emotions > 0 else 0
    print(f"{emotion}: {percentage}%")

cap.release()
cv2.destroyAllWindows()
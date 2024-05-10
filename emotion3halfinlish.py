import cv2
from deepface import DeepFace

# 載入臉部級聯分類器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#  開始擷取影片
cap = cv2.VideoCapture(0)

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

        # 在臉部周圍畫矩形並標記預測的情緒
        cv2.rectangle(frame, (x, y), (x + w, y + h), (160, 32, 240), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 32, 240), 2)

    # 顯示結果框架
    cv2.imshow('chia hao', frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放擷取並關閉所有視窗
cap.release()
cv2.destroyAllWindows()


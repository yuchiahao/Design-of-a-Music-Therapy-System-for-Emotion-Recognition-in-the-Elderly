import cv2
from deepface import DeepFace
import numpy as np

img = cv2.imread('ttteer.jpg')     # 讀取圖片
if img is None:
    print("無法讀取圖片")
else:
    try:
        analyze = DeepFace.analyze(img, actions=['emotion'] )  # 辨識圖片人臉資訊，取出情緒資訊
        print(analyze)
    except:
        pass

    cv2.imshow('hao', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
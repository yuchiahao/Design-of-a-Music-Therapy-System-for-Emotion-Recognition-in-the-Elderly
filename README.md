# 🎵 Emotion-Based Music Healing System｜情緒辨識音樂療癒系統

This is a facial emotion recognition system designed to support music therapy for the elderly, based on the author's master's thesis.  
本系統為一套面向高齡者的臉部情緒辨識音樂療癒工具，源自作者碩士論文開發成果。

> 🎓 Thesis Title 論文名稱：**Music Healing System Design with Emotion Recognition for the Elderly**  
> 🏫 Institute 所屬單位：Mackay Medical College – Graduate Institute of Gerontechnology  
> 👨‍🎓 Author 作者：Yu Chia-Hao（余家豪）  
> 👨‍🏫 Advisor 指導教授：Professor Ming-Cheng Yang（楊明正 教授）  
> 📅 Defense Date 答辯日期：July 22, 2025（民國 113 年）

---

## 📘 Introduction｜系統介紹

This system captures real-time video from the webcam, detects faces using OpenCV, and recognizes facial emotions using DeepFace. When an emotion is detected, it plays a matching piece of music to support emotional healing and well-being.

本系統透過攝影機即時擷取人臉影像，結合 OpenCV 進行臉部偵測，並利用 DeepFace 進行情緒辨識。當辨識出特定情緒後，系統會播放對應的音樂，以達到輔助療癒之效。

> ⚠️ **Note 注意：**  
> Music files are not included in this repository due to copyright reasons.  
> 由於著作權考量，專案中未提供任何音樂檔案。使用者可自備音樂放入 `music/` 對應資料夾。

---

## ✨ Features｜功能特色

- 🎥 Real-time facial detection 即時臉部偵測  
- 😊 Emotion recognition via DeepFace 應用 DeepFace 進行情緒辨識  
- 🎵 Emotion-based music playback 情緒對應音樂播放（需自備音樂）  
- 🖥️ Standalone desktop application 獨立桌面應用程式  
- 💬 Traditional Chinese UI 支援繁體中文顯示（含中文字型）

---

## 📂 Project Structure｜專案架構

.
├── emotion_recognition.py # Main application script 主程式
├── haarcascade_frontalface_default.xml # OpenCV face detection model 臉部辨識模型
├── music/ # Emotion-based music folders 情緒分類音樂資料夾（不含音樂）
├── fonts/ # Chinese fonts 中文字型
├── LICENSE # License 授權條款
└── README.md # This file 本說明文件



---

## ⚙️ Requirements｜執行環境

- Python 3.10+
- OpenCV
- DeepFace
- NumPy
- Pillow (PIL)
- Pygame (for future use, optional)

### Install dependencies｜安裝套件


pip install -r requirements.txt
如果沒有 requirements.txt，你也可以單獨安裝：


pip install opencv-python deepface numpy Pillow pygame
🚀 How to Run｜執行方式
Clone the repository 複製此專案：

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Make sure you have haarcascade_frontalface_default.xml in the directory
確認專案內含有臉部辨識模型檔案（可自 OpenCV GitHub 下載）

Place music files in the following directories 根據情緒放入音樂檔案：


music/happy/
music/sad/
music/angry/
Run the program 啟動系統：


python emotion_recognition.py
Press q to exit 按 q 離開程式。

🧠 System Flow｜系統流程簡述
Webcam captures video 擷取即時影像

OpenCV detects face 偵測人臉

DeepFace analyzes facial emotion DeepFace 辨識情緒

Matching music is played 播放對應音樂

Display emotion label on screen 顯示中文情緒標籤

📜 License｜授權條款
This project is licensed under a modified MIT License for academic and non-commercial use only.
Users must cite the original thesis if used in academic research. Commercial use requires permission.

本專案依據 MIT 授權條款釋出，僅供學術用途與非商業展示。
若用於研究請註明出處，如需商業用途請聯繫作者。

📄 See full terms 詳見 LICENSE 檔案。

🙋 Contact｜聯絡方式
If you would like to collaborate, request permission for commercial use, or provide feedback, please contact:
📧 [dvdbssss@gmail.com] 

🙏 Acknowledgments｜致謝
Professor Ming-Cheng Yang for academic guidance

The open-source community (especially DeepFace and OpenCV)

馬偕醫學院高齡福祉科技研究所之師長與同儕

⭐ Give this repository a star if you find it useful!
⭐ 如果你覺得本專案有幫助，請幫我按下 Star 支持！

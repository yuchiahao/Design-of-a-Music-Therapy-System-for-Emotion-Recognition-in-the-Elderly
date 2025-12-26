# Emotion-Based Music Therapy System (Demo)

本專案為一套結合「臉部情緒辨識」與「音樂即時回饋」的展示系統，
透過攝影機擷取臉部影像，分析使用者當下情緒，並播放對應情緒的音樂，
以驗證情緒辨識技術在互動式應用上的可行性。

本版本為 **Demo**，著重於功能展示與系統流程。

---

## 🎯 功能特色

- 即時臉部偵測（Webcam）
- 臉部表情情緒辨識
- 依辨識結果自動播放對應音樂
- 單機即可執行，適合展示與測試
- Python 撰寫，模組化設計

---

## 🛠 使用技術

- Python
- OpenCV（臉部偵測）
- DeepFace（情緒分析）
- Pygame（音樂播放）
- NumPy / Pillow

---

## 🖥 執行方式

### 1️⃣ 安裝相依套件
```bash
pip install -r requirements.txt

執行展示程式
python emotion_music_demo.py


執行後會啟動攝影機，系統將即時辨識臉部情緒並播放對應音樂。

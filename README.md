# 🎭 Real-Time Emotion Detection with CNN + MediaPipe

This project implements a **real-time facial emotion recognition system** using a custom **Convolutional Neural Network (CNN)** combined with **MediaPipe Face Mesh** for face detection, alignment, and tracking.

It captures live webcam input, processes facial regions, predicts emotions frame-by-frame, and performs a **robust session-level analysis using a tiered confidence point system**.

---

## 🚀 Features

- 📷 Real-time webcam emotion detection  
- 🧠 Custom CNN model (PyTorch)  
- 🧍 Face detection & landmark tracking (MediaPipe)  
- 🔄 Face alignment (rotation correction)  
- 📊 Live probability visualization  
- 🎯 Confidence-based scoring system  
- ⏱️ Session-level emotion analysis  

---

## 🧠 Model Overview

- Input size: `64x64`  
- Architecture:
  - 4 × Conv + ReLU + MaxPool layers  
  - Fully connected layers with Dropout  
- Classes:
  - Angry  
  - Happy  
  - Neutral  
  - Sad  

---

## 📊 Emotion Analysis System

This project uses a **Tiered Confidence Point System** instead of relying on single-frame predictions:

- **< 50% confidence → 0 points**  
- **50% – 95% → 1 point**  
- **> 95% → 3 points**  

The final emotion is determined by the **highest total accumulated points across all frames**, making predictions more stable and reliable.

---

## 🔍 Pipeline

1. Capture webcam frame  
2. Detect face using MediaPipe  
3. Crop and align face  
4. Preprocess image  
5. Run CNN inference  
6. Store prediction data  
7. Perform session-level analysis  

---

## ⚙️ Tech Stack

- PyTorch  
- OpenCV  
- MediaPipe  
- NumPy  
- PIL  
- Torchvision  

---

## 🖥️ Installation

```bash
pip install torch torchvision mediapipe opencv-python numpy pillow

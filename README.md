# 🧠 Binary Video Classification using CNNs  
### Violence Detection in Real-Life Video Clips  

---

## 📘 Overview  
This project focuses on **detecting violence in videos** using **Convolutional Neural Networks (CNNs)**.  
The goal is to accurately classify short video clips as either *violent* or *non-violent*.  

Violence detection is an essential task for applications such as surveillance, content moderation, and public safety monitoring.  

Three different CNN architectures are implemented and evaluated to compare their performance on spatial and spatio-temporal feature learning.  

---

## 📂 Dataset  
**Dataset:** [Real Life Violence and Non-Violence Dataset (Kaggle)](https://www.kaggle.com/datasets/karandeep98/real-life-violence-and-nonviolencedata/data)

- Contains short clips labeled as:
  - 🔴 *Violence*
  - 🟢 *Non-Violence*
- Each video is decomposed into frames for training and testing.
- Most clips contain 5–6 frames, with a few outliers (≈200 frames) discarded during preprocessing.

---

## ⚙️ Models Implemented  

### 🧩 **Model 1 – 2D CNN**  
A custom CNN that processes individual video frames as static images.  
- **Input:** RGB video frames resized to 128×128  
- **Layers:** Convolution → ReLU → MaxPooling → Dense  
- **Output:** Binary classification  
- **Use Case:** Baseline performance using spatial-only information  

---

### 🧠 **Model 2 – ResNet50 (Transfer Learning)**  
A fine-tuned ResNet50 model applied to individual frames for improved feature extraction.  
- **Input:** Frame-wise data (frozen + fine-tuned layers)  
- **Strengths:** Captures deeper spatial features  
- **Use Case:** Enhances generalization for real-world scenes  

---

### 🎥 **Model 3 – 3D CNN**  
A convolutional architecture extended to the temporal domain.  
- **Input:** Video clips as 3D tensors (frames × height × width × channels)  
- **Layers:** 3D Conv → 3D Pool → Dense → Softmax  
- **Use Case:** Captures both spatial and motion-based cues  

---

## 🧪 Training Details  
- **Frameworks:** TensorFlow / Keras, OpenCV, NumPy, Scikit-learn  
- **Split:** 80% training / 20% testing  
- **Normalization:** Frames scaled to [0,1]  
- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy  

---

## 📈 Results Summary  
| Model | Type | Key Features | Expected Performance |
|--------|------|---------------|-----------------------|
| 2D CNN | Baseline | Frame-based | ~80–85% Accuracy |
| ResNet50 | Transfer Learning | Deeper spatial learning | ~85–90% Accuracy |
| 3D CNN | Spatio-temporal | Motion + spatial fusion | ~90–95% Accuracy(When preprocessed proeprly) |

---
101096_CNN_VideoClassification.ipynb

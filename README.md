## 🎧 Audio Feature Extraction

To classify emotions from speech, we extracted a diverse set of **temporal**, **spectral**, and **delta** features using `librosa`. Each feature was summarized using both **mean** and **standard deviation**, resulting in a **556-dimensional feature vector** per audio file.

### 🔍 Extracted Features

#### 🔹 Temporal Features
- **Root Mean Square Energy (RMS)**
- **Zero Crossing Rate (ZCR)**
- **Spectral Centroid**
- **Spectral Bandwidth**
- **Spectral Rolloff**

#### 🔹 Spectral Features
- **MFCCs** – 40 Mel-frequency cepstral coefficients
- **Chroma Frequencies** – 12-dimensional chroma vector
- **Mel Spectrogram** – 128 filterbanks
- **Spectral Contrast** – 7 frequency bands
- **Tonnetz** – 6 tonal centroid features

#### 🔹 Delta Features
- **Delta (Δ MFCC)** – First derivative of MFCCs
- **Delta-Delta (Δ² MFCC)** – Second derivative of MFCCs

Each feature type was summarized across the time axis using:
- **Mean**
- **Standard Deviation**

> 💡 This results in a final vector of **556 features**:  
> `(number of raw features × 2 for mean and std) = 278 × 2 = 556`

---

## ⚙️ Preprocessing Pipeline

Before training, the features were passed through the following preprocessing steps:

1. **Min-Max Scaling**  
   - Normalized features to a [0, 1] range for consistent learning behavior.

2. **Principal Component Analysis (PCA)**  
   - Reduced feature dimensionality while retaining **95% of the original variance**, resulting in **148 principal components** as final model inputs.

---

## 🧠 Model Architecture (Deep Neural Network)

The classification model is a regularized **fully connected deep neural network (DNN)** with the following architecture:

- **Input Layer**: 148 neurons (after PCA)
- **Dense Layer 1**: 256 units + ReLU + BatchNorm + Dropout
- **Dense Layer 2**: 64 units + ReLU + BatchNorm + Dropout
- **Dense Layer 3**: 64 units + ReLU + BatchNorm + Dropout
- **Output Layer**: 8 units + Softmax (for 8 emotion classes)

**Regularization** techniques used:
- `Dropout`: Prevents overfitting
- `Batch Normalization`: Stabilizes and speeds up training

---

## 📊 Model Performance

Trained on the combined **RAVDESS speech and song** dataset, the model achieved:

| Metric              | Value     |
|---------------------|-----------|
| **Accuracy**        | 81.47% ✅ |
| **Macro F1 Score**  | 80.52% ✅ |
| **Micro F1 Score**  | 81.47% ✅ |
| **Per-Class Recall**| ≥ 75% for all classes ✅ |

> ✅ The model meets all target thresholds for overall accuracy, macro F1, and per-class performance.

---

## 📁 Summary

This project combines **advanced audio signal processing** and **deep learning** to classify emotional states from raw `.wav` or `.mp3` files, packaged in a clean and interactive **Streamlit web app**.


import numpy as np
import pandas as pd 
import joblib
from tensorflow.keras.models import load_model
import librosa
import soundfile as sf
def summarize(feature):
    return np.hstack((np.mean(feature, axis=1), np.std(feature, axis=1)))
def extract_features(uploaded):
    y, sr = librosa.load(uploaded, sr=22050)
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
    mel      = librosa.feature.melspectrogram(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz  = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    delta    = librosa.feature.delta(mfcc)
    delta2   = librosa.feature.delta(mfcc, order=2)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth= librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr      = librosa.feature.zero_crossing_rate(y)
    rms      = librosa.feature.rms(y=y)
    
    return np.concatenate([
        summarize(mfcc), summarize(chroma), summarize(mel),
        summarize(contrast), summarize(tonnetz), summarize(delta),
        summarize(delta2), summarize(centroid), summarize(bandwidth),
        summarize(rolloff), summarize(zcr), summarize(rms)
    ]) 
label_map = {
    0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
    4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
}
scaler = joblib.load("Model/scaler.pkl")
pca = joblib.load("Model/pca.pkl")
model = load_model("Model/model.h5")

file_path=r''
x_input=extract_features(file_path).reshape(1,556)
x_scaled = scaler.transform(x_input)
x_pca = pca.transform(x_scaled)
y_pred = np.argmax(model.predict(x_pca), axis=1)
y_mapped = [label_map[y] for y in y_pred]
print(y_mapped[0])

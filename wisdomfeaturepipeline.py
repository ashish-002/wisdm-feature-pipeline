"""
WISDM Sensor Data Preprocessing & Feature Extraction Pipeline
Author: Ashish
Description:
    - Load and clean WISDM sensor data
    - Segment sensor signals
    - Extract statistical and frequency-based features
    - Save features for downstream ML/DL or multimodal fusion
    - Includes placeholders for Deep Learning and Computer Vision extensions
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# 1. Load and Clean Raw Data
# ===============================
def load_and_clean(path=r'data/WISDM_ar_v1.1_raw.txt'):
    df = pd.read_csv(
        path,
        header=None,
        names=['user', 'activity', 'timestamp', 'x', 'y', 'z'],
        on_bad_lines='skip'
    )
    df['z'] = df['z'].str.replace(';', '', regex=False).astype(float)
    df.dropna(inplace=True)
    df['timestamp'] = df['timestamp'].astype(float)
    df = df.sort_values(by=['user', 'timestamp']).reset_index(drop=True)
    return df

# ===============================
# 2. Exploratory Data Analysis
# ===============================
def eda(df):
    print("Unique Activities:", df['activity'].unique())
    plt.figure(figsize=(10,4))
    sns.countplot(x='activity', data=df)
    plt.xticks(rotation=45)
    plt.title("Activity Distribution in WISDM Dataset")
    plt.show()
    sample_user = df[df['user'] == 1]
    sample_walk = sample_user[sample_user['activity'] == 'Walking'][:500]
    plt.figure(figsize=(10,4))
    plt.plot(sample_walk['x'], label='X')
    plt.plot(sample_walk['y'], label='Y')
    plt.plot(sample_walk['z'], label='Z')
    plt.title("Accelerometer Signals - Walking")
    plt.legend()
    plt.show()

# ===============================
# 3. Segment Signal Data
# ===============================
def segment_signal(df, window_size=200, overlap=0.5):
    step = int(window_size * (1 - overlap))
    segments, labels = [], []
    for start in range(0, len(df) - window_size, step):
        window = df.iloc[start:start + window_size]
        if len(window['activity'].unique()) == 1:
            segments.append(window[['x','y','z']].values)
            labels.append(window['activity'].iloc[0])
    return np.array(segments), np.array(labels)

# ===============================
# 4. Extract Features
# ===============================
def extract_features(segments, labels):
    feature_list = []
    for seg in segments:
        feats = []
        for i in range(seg.shape[1]):
            signal = seg[:, i]
            feats += [
                np.mean(signal),
                np.std(signal),
                np.min(signal),
                np.max(signal),
                np.median(signal),
                np.sum(np.abs(np.diff(signal))),
                np.mean(np.fft.fft(signal).real[:10])
            ]
        feature_list.append(feats)
    feature_df = pd.DataFrame(feature_list)
    feature_df['activity'] = labels
    return feature_df

# ===============================
# 5. Save Features
# ===============================
def save_features(features, output_path='results/sensor_features.csv'):
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    features.to_csv(output_path, index=False)
    print(f"✅ Features saved to {output_path}")

# ===============================
# 6. Visualize Features
# ===============================
def visualize_features(features):
    plt.figure(figsize=(10,4))
    sns.boxplot(x='activity', y=features.iloc[:,0], data=features)
    plt.xticks(rotation=45)
    plt.title("Distribution of Feature[0] (mean_x) per Activity")
    plt.show()

    sns.heatmap(features.drop(columns='activity').corr(), cmap='coolwarm', center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()

# ===============================
# 7. Main
# ===============================
if __name__ == "__main__":
    data_path = r'data/WISDM_ar_v1.1_raw.txt'
    df = load_and_clean(data_path)
    print(f"✅ Loaded {len(df)} rows")
    
    eda(df)

    segments, labels = segment_signal(df)
    print(f"✅ Segmented into {len(segments)} windows with {len(labels)} labels")
    
    features = extract_features(segments, labels)
    print(f"✅ Extracted {features.shape[1]-1} features for {features.shape[0]} samples")

    save_features(features)
    visualize_features(features)

    print("✅ Pipeline complete. Features ready for ML/DL or multimodal fusion.")

# ===============================
# 8. Future Work Placeholders
# ===============================

# Deep Learning Example (LSTM / CNN)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# model = Sequential([
#     LSTM(64, input_shape=(200,3)),
#     Dropout(0.3),
#     Dense(6, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=20, batch_size=64)

# Computer Vision Example (OpenCV)
# import cv2
# def extract_optical_flow(video_path, max_frames=100):
#     cap = cv2.VideoCapture(video_path)
#     ret, prev = cap.read()
#     prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#     mag_list = []
#     count = 0
#     while cap.isOpened() and count < max_frames:
#         ret, frame = cap.read()
#         if not ret: break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
#                                             0.5,3,15,3,5,1.2,0)
#         mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
#         mag_list.append(np.mean(mag))
#         prev_gray = gray
#         count +=1
#     cap.release()
#     return np.mean(mag_list), np.std(mag_list)

# Feature-level fusion placeholder
# sensor_feats = pd.read_csv('results/sensor_features.csv')
# video_feats = pd.read_csv('results/video_features.csv')
# fused = pd.concat([sensor_feats.drop('activity',axis=1), video_feats], axis=1)
# fused['activity'] = sensor_feats['activity']
# fused.to_csv('results/fused_features.csv', index=False)
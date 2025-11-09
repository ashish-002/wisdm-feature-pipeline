# WISDM Sensor Data Feature Extraction Pipeline

## 📌 Overview

This repository contains a Python pipeline for preprocessing and feature extraction from the [WISDM Human Activity Recognition Dataset](httpswww.cis.fordham.eduwisdmdataset.php).  
The pipeline is modular, reproducible, and designed as Phase 1 of a larger multimodal project involving sensor and video data.  

It prepares sensor features for downstream machine learning or deep learning tasks such as activity recognition.

---

## 🧩 Features

- Load and clean raw accelerometer data from WISDM
- Segment continuous sensor data into overlapping windows
- Extract statistical and frequency-based features (mean, std, FFT energy, etc.)
- Save features as CSV for MLDL pipelines
- Visualize feature distributions and correlations
- Fully modular and documented Python functions

---

## ⚡ Future Work  Extensions

The pipeline is designed to easily integrate the following

1. Deep Learning for Time-Series
   - LSTM, CNN, or Transformer-based activity recognition
   - Sequence modeling using segmented sensor windows

2. Computer Vision  Video Features
   - Optical flow, CNN embeddings, or pretrained feature extractors from MP4 videos
   - Feature-level fusion with sensor data

3. Multimodal Fusion
   - Combine sensor + video features for improved activity recognition
   - Ready for feature-level or decision-level fusion

---

## 🛠 Installation

### Requirements

- Python 3.9+
- Packages `pandas`, `numpy`, `matplotlib`, `seaborn`, `opencv-python` (optional for CV)
- Optional for DL `tensorflow` or `torch`

```bash
pip install pandas numpy matplotlib seaborn opencv-python tensorflow

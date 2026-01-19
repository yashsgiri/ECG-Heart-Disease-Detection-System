# ECG Heart Disease Detection System  
**Real-Time ECG Analysis using Signal Processing and Machine Learning**

---

## 📌 Project Overview

This project implements a **real-time ECG-based heart disease detection system** using **Python, signal processing, and machine learning**.  
The system acquires ECG signals from a patient through a sensor and Arduino-based setup, preprocesses the signals, extracts clinically relevant features, and classifies the ECG as **Normal** or **Abnormal**.  
A detailed **PDF medical report** is automatically generated for analysis.

This project focuses on **practical implementation**, integrating hardware, signal processing, and ML techniques.

---

## 🎯 Objectives

- Acquire real-time ECG signals using hardware sensors  
- Preprocess ECG signals to remove noise  
- Extract heart rate and HRV features  
- Classify ECG signals using machine learning  
- Generate automated ECG analysis reports  

---

## 🏗️ System Architecture

**Workflow:**

1. ECG signal acquisition from patient  
2. Analog signal transmission to Arduino Uno  
3. Digital signal transfer to PC via serial communication  
4. ECG preprocessing and noise filtering  
5. Feature extraction (Heart Rate & HRV)  
6. Machine learning–based classification  
7. Automated PDF report generation  

*(Architecture diagram can be added here in future)*

---

## ⚙️ Hardware Components

- ECG Sensor Module (AD8232 / ADS8230 / ADS8231 compatible)  
- Arduino Uno  
- ECG Electrodes  
- USB Serial Communication  

---

## 🧪 Software & Technologies Used

- **Programming Language:** Python  
- **Signal Processing:** NumPy, SciPy  
- **Machine Learning:** Scikit-learn (Random Forest)  
- **Data Handling:** Pandas  
- **Visualization:** Matplotlib  
- **GUI:** Tkinter  
- **Report Generation:** ReportLab  
- **Hardware Communication:** PySerial  

---

## 🔍 Key Features

- Real-time ECG signal acquisition  
- ECG signal preprocessing using bandpass filtering  
- R-peak detection and heart rate calculation  
- Heart Rate Variability (HRV) feature extraction:
  - Beats Per Minute (BPM)
  - SDNN
  - RMSSD
- Automatic ECG labeling based on clinical thresholds  
- Machine learning–based ECG classification  
- GUI-based ECG visualization  
- Automated PDF ECG report generation  

---

## 🤖 Machine Learning Approach

- **Model Used:** Random Forest Classifier  
- **Input Features:** Heart rate and HRV metrics  
- **Classification Task:**  
  - Normal ECG  
  - Abnormal ECG  

The model is trained using extracted ECG features and rule-based auto-labeling to perform supervised learning.

---

## 📂 Project Structure

ECG-Heart-Disease-Detection-System/
│
├── data/ # ECG CSV / Excel data files
├── models/ # Trained ML model (.pkl)
├── reports/ # Generated ECG PDF reports
├── src/ # Main Python application
│ └── ecg_main_app.py
│
├── README.md
└── requirements.txt


---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt

python src/ecg_main_app.py

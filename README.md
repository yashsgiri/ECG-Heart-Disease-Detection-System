📌 Project Overview

This project presents a real-time ECG-based heart disease detection system developed using Python, signal processing techniques, and machine learning. The system acquires ECG signals from a patient using a hardware sensor setup, processes the signals in real time, extracts clinically relevant features, and classifies the ECG as Normal or Abnormal. A detailed PDF medical report is generated for further analysis.

The project focuses on practical implementation, combining hardware integration, ECG signal analysis, and ML-based classification.

🧠 Motivation

Cardiovascular diseases are one of the leading causes of mortality worldwide. Early detection using ECG signals can significantly improve diagnosis and treatment outcomes. This project aims to demonstrate how real-time ECG signals can be analyzed and classified using machine learning, making cardiac screening more accessible and automated.

🏗️ System Architecture

Workflow:

ECG signal acquisition from patient using ECG sensor

Analog ECG signal transmission to Arduino Uno

Conversion to digital signal and transfer to PC

ECG signal preprocessing and noise filtering

Feature extraction (Heart Rate & HRV metrics)

Machine Learning–based classification

Automated PDF report generation

⚙️ Hardware Components

ECG Sensor Module (ADS8230 / ADS8231 / AD8232 compatible)

Arduino Uno

USB Serial Communication

ECG Electrodes

🧪 Software & Technologies Used

Programming Language: Python

Signal Processing: NumPy, SciPy

Machine Learning: Scikit-learn (Random Forest)

Data Handling: Pandas

Visualization: Matplotlib

GUI: Tkinter

Report Generation: ReportLab

Hardware Communication: PySerial

🔍 Key Features

Real-time ECG signal acquisition via Arduino

ECG signal preprocessing using bandpass filtering

Peak detection for heart rate calculation

Heart Rate Variability (HRV) feature extraction:

BPM (Beats Per Minute)

SDNN

RMSSD

Automatic ECG labeling based on clinical thresholds

Machine Learning–based ECG classification

GUI-based ECG visualization

Automated PDF report generation

🤖 Machine Learning Approach

Model Used: Random Forest Classifier

Features: Heart rate and HRV metrics

Classification Task:

Normal ECG

Abnormal ECG

The ML model is trained using extracted ECG features and rule-based auto-labeling for supervised learning.

[반도체 소자 이상 탐지 AI 경진대회](https://dacon.io/competitions/official/236224/overview/description)

# Dacon Anomaly Detection Baseline 
This repository contains a baseline for an anomaly detection task using the Dacon dataset.  
Future updates will include model ensemble functionality.

# Overview
### Backbone Model: ResNet18
- The penultimate features of ResNet18 are used to extract meaningful representations from the data.
### Classifier: IsolationForest
- An unsupervised learning algorithm to detect anomalies based on the extracted features.

# Future Updates
### Model Ensemble
- A model ensemble function will be added to improve detection performance.

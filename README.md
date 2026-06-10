# Network Intrusion Detection System using CNN, GNN, and Transformers

## Overview

This project presents an intelligent Network Intrusion Detection System (NIDS) that leverages deep learning models to classify network traffic into:

- Normal Traffic
- Denial of Service (DoS) Attacks
- Probe / Port Scan Attacks

The system evaluates and compares three powerful deep learning architectures:

- Convolutional Neural Network (CNN)
- Graph Neural Network (GNN)
- Transformer

A Streamlit-based web application is integrated for real-time intrusion detection and interactive traffic analysis. 
The project is based on the CIC IDS 2017 dataset and aims to provide an accurate, scalable, and user-friendly cybersecurity solution.

## Features

- Network traffic classification into Normal, DoS, and Probe categories
- Deep learning-based threat detection
- Comparative evaluation of CNN, GNN, and Transformer models
- Automated preprocessing pipeline
- Real-time prediction through Streamlit dashboard
- Confidence score visualization
- Interactive model selection
- Manual feature input support
- Batch CSV/Excel file analysis


## Project Architecture

Raw Network Traffic
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Standardization
        │
        ▼
Model Training
 ┌─────────┬─────────┬─────────┐
 │   CNN   │   GNN   │Transformer│
 └─────────┴─────────┴─────────┘
        │
        ▼
Model Evaluation
        │
        ▼
Streamlit Web Application
        │
        ▼
Real-Time Intrusion Detection



# Lip Reading Project Using Deep Learning

This repository contains the code for a lip-reading project using deep learning. The project aims to develop a model that can interpret lip movements from videos and perform speech recognition. The model utilizes Conv3D and Bidirectional LSTM layers to accurately recognize speech patterns.

## Contents

- [Introduction](#introduction)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Streamlit App](#streamlit-app)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

The ability to understand speech by observing lip movements can be crucial in scenarios where audio is not clear or unavailable. This project focuses on developing a deep learning model that can effectively interpret lip movements and perform speech recognition. The model is trained on a dataset of videos with corresponding alignments.

## Data

The dataset used for training the lip-reading model consists of videos and alignment files. The videos contain individuals speaking, and the alignment files provide information about the corresponding phonemes. The `load_video` function is responsible for loading the video frames, while the `load_alignments` function processes the alignment files.

## Model Architecture

The lip-reading model employs a Conv3D and Bidirectional LSTM architecture. It consists of multiple Conv3D layers to capture spatial and temporal features from the video frames. The Bidirectional LSTM layers allow the model to learn patterns in both forward and backward directions. The final dense layer, followed by a softmax activation, outputs the predicted phoneme sequence.

## Training

The model is trained using the provided dataset. The data pipeline is set up using TensorFlow's Dataset API, allowing efficient preprocessing and loading of the video frames and alignments. The `CTCLoss` function is used as the loss function for training the model. A learning rate scheduler is implemented to adjust the learning rate during training.

## Evaluation

The trained model is evaluated using a separate test set. The performance of the model is assessed by comparing its predictions against the ground truth phoneme sequences. The accuracy and effectiveness of the lip-reading system are measured based on how well the predicted phoneme sequences match the actual phonemes.

## Streamlit App

A Streamlit app has been developed to demonstrate the lip-reading system. The app provides a user interface for select a video and visualizing the lip-reading results. It utilizes the trained model to process the videos and display the predicted phoneme sequences. The Streamlit app enhances the accessibility and usability of the lip-reading system.

## Installation

To install the required dependencies, use the following command:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```
git clone https://github.com/Tayyab885/Lip-Reading-Project-Using-Deep-Learning
```

2. Install the dependencies as mentioned in the installation section.

3. Run the Streamlit app:

```
streamlit run app.py
```

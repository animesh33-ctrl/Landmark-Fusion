# Sign Language Recognition System

A deep learning-based sign language recognition system using keypoint extraction and transformer/CNN models for both word-level and sentence-level predictions.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Training](#training)
- [Inference](#inference)

## Features

- **Multi-level Recognition**: Word-level and sentence-level sign language prediction
- **Multiple Architectures**: CNN, LSTM, and Transformer models
- **Keypoint Extraction**: Advanced pose and hand landmark detection
- **Real-time Inference**: Live video prediction capabilities
- **Pre-trained Checkpoints**: Ready-to-use model weights

## Project Structure

```
Sign Language 5/
├── config/              # Configuration files
├── src/                 # Source code
│   ├── dataset_loader.py
│   ├── keypoint_extractor.py
│   ├── preprocessing.py
│   └── models/         # Model implementations
│       ├── cnn_model.py
│       ├── lstm_model.py
│       └── transformer_model.py
├── training/           # Training scripts
│   ├── train_word_model.py
│   └── train_sentence_model.py
├── inference/          # Inference modules
│   └── realtime_prediction.py
├── utils/              # Utility functions
│   ├── metrics.py
│   └── path_checker.py
├── checkpoints/        # Pre-trained models
└── requirements.txt    # Dependencies
```

## Installation

1. Clone the repository and navigate to the project directory:

   ```bash
   cd "Sign Language 5"
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Main Entry Point

Run the main application:

```bash
python main.py
```

### Configuration

Edit model and training parameters in `config/config.py`

## Models

- **CNN Model** (`cnn_model.py`): Convolutional Neural Network for word recognition
- **LSTM Model** (`lstm_model.py`): Long Short-Term Memory for sequence processing
- **Transformer Model** (`transformer_model.py`): Transformer architecture for sentence-level recognition

## Training

### Train Word-Level Model

```bash
python training/train_word_model.py
```

### Train Sentence-Level Model

```bash
python training/train_sentence_model.py
```

## Inference

### Real-time Prediction

```bash
python inference/realtime_prediction.py
```

## Pre-trained Checkpoints

Available models in `checkpoints/`:

- `best_cnn_word.pth` - Best CNN model for word recognition
- `best_transformer_sentence.pth` - Best Transformer model for sentence recognition
- `last_cnn_word.pth` - Last trained CNN word model
- `last_transformer_sentence.pth` - Last trained Transformer sentence model

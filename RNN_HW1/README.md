# RNN_HW1
## Project Summary

This project implements and trains an LSTM (Long Short-Term Memory) neural network model to distinguish between AI-generated and human-written text. The model is built using PyTorch and trained on the [AI vs Human Text dataset](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) from Kaggle, which contains samples of text written by humans and AI systems.

The LSTM architecture was chosen for its ability to capture sequential patterns and long-range dependencies in text data, making it well-suited for this text classification task.

## Requirements

To run this project, you need the following Python packages:

```
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
rich==13.9.4
pandas==2.2.3
scikit-learn==1.6.1
```

You can install these packages using pip:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 rich==13.9.4 pandas==2.2.3 scikit-learn==1.6.1
```

Or if you use conda:

```bash
conda install pytorch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 rich==13.9.4 pandas==2.2.3 scikit-learn==1.6.1 -c pytorch
```
## Training and Testing

To train and test the LSTM model, use the `LSTM_Train.py` script with the appropriate arguments:

```bash
python3 LSTM_Train.py --test_model --csv_path /home/aicv/work/AI_Human.csv --log_path runs/lstm_experiment --best_model_path ./models/lstm_model_best.pth --final_model_path ./models/lstm_model.pth --test_model_path ./models/lstm_model.pth --write_result_path ./runs/test/test_results.txt --model_folder ./models --model_name lstm_model
```

Alternatively, you can use the [RNN_HW1.ipynb](RNN_HW1.ipynb) notebook which contains the implementation and results for testing and training the LSTMClassifier model with 2 layers (num_layers=2).

### Parameters Explanation:

- `--test_model`: Flag to indicate testing mode
- `--csv_path`: Path to the dataset CSV file
- `--log_path`: Directory to save training logs
- `--best_model_path`: Path to save the best model during training
- `--final_model_path`: Path to save the final trained model
- `--test_model_path`: Path to the model for testing
- `--write_result_path`: File path to save test results
- `--model_folder`: Directory to store model files
- `--model_name`: Name prefix for the model files


## Model Architecture

This project implements and compares four different RNN-based models for text classification:

### 1. LSTMClassifier
A basic LSTM model that processes sequential text data through:
- Embedding layer to convert token IDs to dense vectors
- LSTM layer to capture sequential patterns
- Linear layer for final classification

### 2. ConvLSTMClassifier
A hybrid model that combines convolutional layers with LSTM:
- Embedding layer for token representation
- 1D convolutional layers to extract local features
- LSTM layer to capture long-range dependencies 
- Linear layer for final classification

### 3. BiStackedLSTMClassifier
A more complex model that uses bidirectional stacked LSTMs:
- Embedding layer for input representation
- Stacked bidirectional LSTM layers
- Linear layer for final classification

### 4. AttenationBiStackedLSTMClassifier
The most sophisticated model with attention mechanism:
- Embedding layer for token representation
- Stacked bidirectional LSTM layers
- Attention mechanism to focus on the most relevant parts of the text
- Linear layer for final classification

## Results

A detailed analysis of the model performance, including training curves, confusion matrices, and in-depth comparisons between different architectures, is available in the [RNN_HW1_Report.pdf](RNN_HW1_Report.pdf) document.

The table below shows the performance summary of each model architecture on the test dataset:

| # | Model Name | Configuration | Accuracy |
|---|------------|--------------|----------|
| 1 | 2LSTM | LSTMClassifier (num_layers=2) | 0.9960 |
| 2 | LSTM | LSTMClassifier (num_layers=1) | 0.9973 |
| 3 | CNN+LSTM | ConvLSTMClassifier (num_layers=1) | 0.9813 |
| 4 | CNN+2LSTM | ConvLSTMClassifier (num_layers=2) | 0.9987 |
| 5 | BiStackedLSTM | BiStackedLSTMClassifier (num_layers=2) | 0.9994 |
| 6 | AttentionBiStackedLSTM | AttenationBiStackedLSTMClassifier (num_layers=2) | 0.9993 |

The BiStackedLSTM model achieved the highest accuracy at 0.9994, closely followed by the AttentionBiStackedLSTM model with 0.9993 accuracy. For comprehensive evaluation metrics and analysis, please refer to the RNN_HW1_Report.pdf document.



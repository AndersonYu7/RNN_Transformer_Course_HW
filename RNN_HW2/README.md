# RNN_HW1
## Project Summary

This project implements and trains an LSTM (Long Short-Term Memory) neural network model using PyTorch to classify tweets as either related or unrelated to real disasters. The dataset utilized is "Natural Language Processing with Disaster Tweets" from Kaggle [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview), containing tweet samples labeled as disaster-related (1) or not (0).

The objective of this project includes:

Developing two predictive models—one based on an LSTM architecture and another based on a GRU (Gated Recurrent Unit).

Evaluating and comparing the performance of both models, including aspects such as training accuracy, training speed, and memory usage.

Generating predictions using the provided testing data and preparing a CSV file formatted according to the required submission guidelines.

Compiling and submitting a comprehensive report detailing the methodologies, results, and comparative analysis of model performance.

All code, predictive outputs, and the analytical report are organized and hosted on GitHub, providing easy access and review.

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

Alternatively, you can use the [RNN_HW1.ipynb](RNN_HW1.ipynb) notebook which contains the implementation and results for 

Developing multiple predictive models including:
* [LSTM.ipynb](LSTM.ipynb)

* [LSTM_Dropout.ipynb](LSTM_Dropout.ipynb)

* [2LSTM_Dropout.ipynb](2LSTM_Dropout.ipynb)

* [GRU.ipynb](GRU.ipynb)

* [GRU_Dropout.ipynb](GRU_Dropout.ipynb)

* [2GRU_Dropout.ipynb](2GRU_Dropout.ipynb)

## Results

A detailed analysis of the model performance, including training curves, confusion matrices, and in-depth comparisons between different architectures, is available in the [RNN_HW2_Report.pdf](RNN_HW2_Report.pdf) document.

The table below shows the performance summary of each model architecture on the test dataset from Kaggle:

| # | Model Name | Accuracy |
|---|------------|----------|
| 1 | Lstm | 0.79313 |
| 2 | Lstm dropout | 0.79589 |
| 3 | 2 Lstm dropout | 0.79037 |
| 4 | GRU | 0.78700 |
| 5 | GRU dropout | 0.79681 |
| 6 | 2 GRU dropout | 0.79129 |



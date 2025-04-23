# RNN_HW3
## Project Summary

This project focuses on Named Entity Recognition (NER) using the DNTRI dataset, which contains cybersecurity-related text annotated with various entity types such as organization, tool, area, and attack methods. The goal is to train and evaluate different NER models to identify these entities from raw text.

We implemented and compared two approaches:

- **DistilBERT-based NER model**
    - Achieves strong performance with fewer parameters.
- **SecBERT + LSTM + CRF**
    - Leverages domain-specific embeddings (SecBERT).
    - Uses LSTM for sequence modeling.
    - Applies CRF to improve label dependencies.

The dataset is formatted in CoNLL-style `.txt` files (`train.txt`, `valid.txt`, `test.txt`), where each line contains a token and its corresponding label, and sentences are separated by blank lines.

The project includes data preprocessing, tokenization, model training, evaluation, and output generation for submission. Evaluation metrics such as precision, recall, and F1-score are used to compare model performance across validation and test sets.

## Training and Testing
To train and test the models, use the following notebooks:

- [HW3.ipynb](HW3.ipynb): Implements the DistilBERT-based NER model.
- [HW3_secbert_lstm_crf.ipynb](HW3_secbert_lstm_crf.ipynb): Implements the SecBERT + LSTM + CRF model.

Each notebook contains code for data loading, preprocessing, model training, evaluation, and generating predictions. Follow the instructions within each notebook to reproduce the results.

## Results

A detailed analysis of the model performance, including training curves, confusion matrices, and in-depth comparisons between different architectures, is available in the [RNN_HW3_Report.pdf](RNN_HW3_Report.pdf) document.

The prediction outputs for the test set are provided for both models:

- **DistilBERT-based NER model:** See `predicted_test.txt`
- **SecBERT + LSTM + CRF model:** See `predicted_test_secbert_lstm_crf.txt`

The following table summarizes the evaluation metrics for the two models on the test set:

| Model                          | Precision | Recall   | F1 Score | Accuracy |
|---------------------------------|-----------|----------|----------|----------|
| DistilBERT-based NER            | 0.86684   | 0.88460  | 0.87563  | 0.94914  |
| SecBERT + LSTM + CRF          | 0.79913   | 0.85976  | 0.82834  | 0.95421  |
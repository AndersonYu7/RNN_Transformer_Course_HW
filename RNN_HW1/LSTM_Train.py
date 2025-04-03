import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from DataProcess import DataLoaderFactory
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter  
import argparse

class LSTMClassifier(nn.Module):
    """
    # This LSTMClassifier uses an embedding layer followed by a multi-layer LSTM for text classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # Check if sequence length is zero and return zeros if so.
        if text.size(1) == 0:
            return torch.zeros(text.size(0), self.fc.out_features, device=text.device)
        # Convert input word indices to embeddings
        embedded = self.embedding(text)
        # Process embeddings through LSTM
        _, (hidden, _) = self.lstm(embedded)
        # Use the last hidden state for classification
        return self.fc(hidden[-1])
    
class ConvLSTMClassifier(nn.Module):
    """
    The ConvLSTMClassifier combines a convolutional layer with an LSTM for text classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, num_layers=1):
        super().__init__()
        # Create an embedding layer to convert word indices to dense vectors.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # Create a 1D convolutional layer to capture local features from embeddings.
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1)
        # Create an LSTM layer to capture sequential dependencies.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # Final linear layer to produce the output logits.
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # Handle empty texts by returning zeros output.
        if text.size(1) == 0:
            return torch.zeros(text.size(0), self.fc.out_features, device=text.device)
        # Obtain word embeddings: shape [batch_size, seq_length, embedding_dim]
        embedded = self.embedding(text)
        # Rearrange dimensions for Conv1d: from [batch_size, seq_length, embedding_dim]
        # to [batch_size, embedding_dim, seq_length]
        conv_input = embedded.permute(0, 2, 1)
        # Apply convolution followed by ReLU activation: shape remains [batch_size, embedding_dim, seq_length]
        conv_out = torch.relu(self.conv(conv_input))
        # Rearrange back to LSTM input shape: [batch_size, seq_length, embedding_dim]
        conv_out = conv_out.permute(0, 2, 1)
        # Feed the convolution output into the LSTM; get the hidden states.
        _, (hidden, _) = self.lstm(conv_out)
        # Use the hidden state of the last LSTM layer for classification.
        return self.fc(hidden[-1])
    
class Attention(nn.Module):
    """
    Attention module that computes a context vector.
    It uses the current hidden state as the query and the encoder outputs as keys to 
    calculate compatibility scores. The scores undergo a softmax to produce attention 
    weights, which are then used to compute a weighted sum of the encoder outputs.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Linear layer to transform the query (hidden state)
        self.W = nn.Linear(hidden_dim, hidden_dim)
        # Linear layer to transform the encoder outputs (keys)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        # Learnable parameter for computing the compatibility score
        self.v = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_dim)
        # encoder_outputs: (batch_size, seq_length, hidden_dim)

        # Expand hidden to (batch_size, 1, hidden_dim) to use it as the query for attention
        hidden = hidden.unsqueeze(1)

        # Compute intermediate scores by applying a tanh activation on the sum of
        # transformed hidden (query) and encoder outputs (keys); shape: (batch_size, seq_length, hidden_dim)
        score = torch.tanh(self.W(hidden) + self.U(encoder_outputs))

        # Compute raw attention scores by taking the dot product with the learnable vector v;
        # resulting shape: (batch_size, seq_length)
        attention_weights = torch.matmul(score, self.v)

        # Apply softmax to obtain a probability distribution over the sequence length
        attention_weights = F.softmax(attention_weights, dim=1)

        # Compute the context vector as the weighted sum of encoder outputs according to the attention weights;
        # resulting shape: (batch_size, hidden_dim)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * encoder_outputs, dim=1)
        return context_vector
    
class BiStackedLSTMClassifier(nn.Module):
    """
    BiStackedLSTMClassifier implements a bidirectional LSTM classifier.
    It applies an embedding layer followed by a bidirectional LSTM and a fully connected layer.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, num_layers=2):
        super().__init__()
        # Embedding layer to convert input token indices into dense vectors.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # Bidirectional LSTM layer with the specified number of layers.
        # batch_first=True means the input shape is [batch_size, sequence_length, embedding_dim].
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=True)
        # Fully connected layer that maps the concatenated hidden states to the output dimension.
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        # If the input text has zero length, return a tensor of zeros with appropriate shape.
        if text.size(1) == 0:
            return torch.zeros(text.size(0), self.fc.out_features, device=text.device)
        # Convert input token indices into embeddings.
        embedded = self.embedding(text)
        # Pass embeddings through the bidirectional LSTM.
        # The output is ignored as we are only interested in the hidden states.
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate the last forward and backward hidden states.
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # Pass the concatenated hidden states through the fully connected layer to get logits.
        return self.fc(hidden_cat)

class AttenationBiStackedLSTMClassifier(nn.Module):
    """
    AttenationBiStackedLSTMClassifier implements a bidirectional LSTM classifier augmented with an attention mechanism.
    It utilizes an embedding layer, a bidirectional LSTM to capture both forward and backward context, and an attention module to combine the encoded sequence information.
    The resulting context vector is then passed through a fully connected layer to produce the final output for classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, num_layers=2):
        super().__init__()
        # Initialize the embedding layer to convert input indices to dense vectors.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # Create a bidirectional LSTM layer.
        # Note: This generates outputs with dimension hidden_dim * 2 due to bidirectionality.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=True)
        # Fully connected layer to map the context vector from attention to the desired output dimension.
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # Attention module that operates on the concatenated hidden states (from both directions).
        self.attention = Attention(hidden_dim * 2)

    def forward(self, text):
        # Handle empty input: if the sequence length is zero, return a tensor of zeros.
        if text.size(1) == 0:
            return torch.zeros(text.size(0), self.fc.out_features, device=text.device)
        # Obtain embeddings from the input text.
        embedded = self.embedding(text)
        
        # Process the embeddings through the bidirectional LSTM.
        # outputs: tensor of shape [batch_size, seq_length, hidden_dim*2]
        # hidden: tensor of shape [num_layers*2, batch_size, hidden_dim]
        outputs, (hidden, _) = self.lstm(embedded)
        
        # Concatenate the last hidden state from the forward and backward passes.
        # hidden[-2, :, :] corresponds to the forward LSTM of the last layer.
        # hidden[-1, :, :] corresponds to the backward LSTM of the last layer.
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # Use the concatenated hidden state as query and the LSTM outputs as keys/values for attention.
        context = self.attention(hidden_cat, outputs)
        
        # Map the attention context vector to the output logits.
        return self.fc(context)

def train(model, iterator, optimizer, criterion, device, current_epoch, total_epochs):
    # Set the model to training mode
    model.train()
    epoch_loss = 0

    # Iterate over batches in the training data
    for texts, labels in track(iterator, description=f"[bold][cyan]Epoch {current_epoch}/{total_epochs}[/bold]"):
        # Move the texts and labels to the specified device (CPU or GPU)
        texts, labels = texts.to(device), labels.to(device)

        # Zero out gradients to prevent accumulation
        optimizer.zero_grad()

        # Perform a forward pass through the model
        predictions = model(texts).squeeze(1)

        # Calculate the loss between predictions and actual labels
        loss = criterion(predictions, labels)

        # Perform a backward pass to compute gradients
        loss.backward()

        # Update the model parameters using the computed gradients
        optimizer.step()

        # Accumulate the batch loss
        epoch_loss += loss.item()

    # Return the average loss for the epoch
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    # This function evaluates the model's performance on the given data iterator (e.g., validation set)
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in track(iterator, description="[bold][yellow]Validation[/bold]"):
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts).squeeze(1)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()

            # Apply sigmoid to logits, round predictions, and compare with actual labels
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct += (rounded_preds == labels).sum().item()
            total += labels.size(0)

    return epoch_loss / len(iterator), correct / total

def test_best_model(model, test_loader, criterion, device, model_path="./models/model_best.pth", write_path="./runs/test/test_results.txt"):
    # Load the saved best model weights from disk.
    best_model_path = model_path
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode.
    
    total_loss = 0
    correct = 0
    total = 0
    # Evaluate the model on the test dataset.
    with torch.no_grad():
        for texts, labels in track(test_loader, description="[bold][yellow]Testing[/bold]"):
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts).squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Convert model outputs to probabilities then round them to obtain final predictions.
            preds = torch.round(torch.sigmoid(outputs))
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # Calculate average loss and accuracy.
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    print(f'Final Test Loss: {avg_loss:.4f}, Final Test Accuracy: {accuracy:.4f}')
    
    # Append the test results to a file.
    with open(write_path, 'a') as f:
        f.write(f'\nFinal Test Loss: {avg_loss:.4f}, Final Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_model', action='store_true', help='Enable testing after training')
    parser.add_argument('--csv_path', type=str, default='/home/aicv/work/AI_Human.csv', help='CSV file path')
    parser.add_argument('--log_path', type=str, default='runs/lstm_experiment6', help='TensorBoard log directory')
    parser.add_argument('--best_model_path', type=str, default='./models/lstm_model6_best.pth', help='Path to save the best model')
    parser.add_argument('--final_model_path', type=str, default='./models/lstm_model6.pth', help='Path to save the final model')
    parser.add_argument('--test_model_path', type=str, default='./models/lstm_model6.pth', help='Model path used for testing')
    parser.add_argument('--write_result_path', type=str, default='./runs/test/test_results.txt', help='Path to write test results')
    parser.add_argument('--model_folder', type=str, default='./models', help='Folder to save models')
    parser.add_argument('--model_name', type=str, default='lstm_model6', help='Model name prefix')
    # Note: 'vocab' is defined later, so parameters depending on it will be set when instantiating the model.
    parser.add_argument('--embedding_dim', type=int, default=100, help='Embedding dimension for words')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for the LSTM')
    parser.add_argument('--output_dim', type=int, default=1, help='Number of output units')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    args = parser.parse_args()

    test_model = args.test_model
    csv_path = args.csv_path
    log_path = args.log_path
    best_model_path = args.best_model_path
    final_model_path = args.final_model_path
    test_model_path = args.test_model_path
    write_result_path = args.write_result_path
    model_folder = args.model_folder
    model_name = args.model_name
    
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    output_dim = args.output_dim
    num_layers = args.num_layers
    
    # Create directories if they do not exist
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    data_loader_factory = DataLoaderFactory(csv_path, batch_size_train=64, batch_size_val=64)
    train_loader = data_loader_factory.train_loader
    test_loader = data_loader_factory.test_loader
    val_loader = data_loader_factory.val_loader
    
    vocab = data_loader_factory.vocab  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # switch your Classifier # LSTMClassifier ConvLSTMClassifier BiStackedLSTMClassifier AttenationBiStackedLSTMClassifier
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        padding_idx=vocab['<pad>'],
        num_layers=num_layers
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter(log_path)

    best_acc = 0.0
    total_epochs = 50
    # Training loop
    for epoch in range(1, total_epochs+1):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, total_epochs)
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}')
        
        test_loss, test_acc = evaluate(model, val_loader, criterion, device)
        print(f'Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Test', test_acc, epoch)
        
        # Save best model based on test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved at epoch {epoch} with Test Acc: {test_acc:.4f}')

        # Save model every 10 epochs with a custom name
        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_folder + f'/{model_name}_{epoch}.pth')
            print(f'Model saved at epoch {epoch} as {model_name}_{epoch}.pth')

    writer.close()

    # Save the final model
    torch.save(model.state_dict(), final_model_path)
    
    if test_model:
        test_best_model(model, test_loader, criterion, device, model_path=test_model_path, write_path=write_result_path)

# python3 LSTM_Train.py --test_model --csv_path /home/aicv/work/AI_Human.csv --log_path runs/lstm_experiment7 --best_model_path ./models/lstm_model7_best.pth --final_model_path ./models/lstm_model7.pth --test_model_path ./models/lstm_model7.pth --write_result_path ./runs/test/test_results.txt --model_folder ./models --model_name lstm_model7

# Model names and their corresponding numbers
# 1: LSTMClassifier
# 2: ConvLSTMClassifier
# 3: BiStackedLSTMClassifier
# 4: AttenationBiStackedLSTMClassifier

# Model names and their corresponding numbers
# 1: 2LSTM                  (LSTMClassifier: num_layers=2)                      Acc: 0.9960
# 2: LSTM                   (LSTMClassifier: num_layers=1)                      Acc: 0.9973
# 3: CNN+LSTM               (ConvLSTMClassifier: num_layers=1)                  Acc: 0.9813
# 4: CNN+2LSTM              (ConvLSTMClassifier: num_layers=2)                  Acc: 0.9987
# 5: BiStackedLSTM          (BiStackedLSTMClassifier: num_layers=2)             Acc: 0.9994
# 6: AttentionBiStackedLSTM (AttenationBiStackedLSTMClassifier: num_layers=2)   Acc: 0.9993
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

def tokenizer(text):
    # Split the text into words based on whitespace
    return text.split()

def numericalize(text, vocab):
    # Convert text into a list of numbers according to the given vocabulary.
    # If a word is not in the vocabulary, the '<unk>' token index is used.
    return [vocab.get(word, vocab['<unk>']) for word in tokenizer(text)]

def pad_collate_fn(batch, pad_idx):
    # Custom collate function to pad sequences in a batch to the same length.
    texts, labels = zip(*batch)
    text_lens = [len(text) for text in texts]
    max_len = max(text_lens)
    # Create a tensor filled with pad index values.
    padded_texts = torch.full((len(texts), max_len), pad_idx, dtype=torch.long)
    for i, text in enumerate(texts):
        padded_texts[i, :text_lens[i]] = text
    return padded_texts, torch.tensor(labels, dtype=torch.float)

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        # Initialize dataset with texts, labels and vocabulary
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.texts)

    def __getitem__(self, idx):
        # Retrieve the numericalized text and corresponding label for a given index
        text = numericalize(self.texts[idx], self.vocab)
        label = self.labels[idx]
        return torch.tensor(text), torch.tensor(label, dtype=torch.float)
    
class DataLoaderFactory:
    def __init__(self, csv_path, text_col='text', label_col='generated',
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42,
                 batch_size_train=32, batch_size_val=32, batch_size_test=1):
        # Initialize factory with file path and various ratios for training, validation, and test sets.
        self.csv_path = csv_path
        self.text_col = text_col
        self.label_col = label_col

        # Check if the sum of ratios for train, validation, and test equals 1
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
            raise ValueError("Train, validation and test ratios must sum to 1.0")
            
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test

        self.vocab = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Prepare data and create dataloaders
        self._prepare_data()

    def _build_vocab(self, texts):
        # Build a vocabulary from the texts using a Counter to count word frequencies.
        counter = Counter()
        for text in texts:
            counter.update(tokenizer(text))
        # Create vocabulary mapping starting from index 2 
        # (reserve index 0 for <unk> and index 1 for <pad>)
        vocab = {word: i+2 for i, (word, _) in enumerate(counter.items())}
        vocab['<unk>'] = 0
        vocab['<pad>'] = 1
        return vocab

    def _prepare_data(self):
        # Read the dataset from the CSV file.
        df = pd.read_csv(self.csv_path)
        texts = df[self.text_col].tolist()
        labels = df[self.label_col].tolist()

        # Build vocabulary from all texts.
        self.vocab = self._build_vocab(texts)

        # Split dataset into training and combined validation+test sets.
        texts_train, texts_val_test, labels_train, labels_val_test = train_test_split(
            texts, labels, test_size=self.test_ratio+self.val_ratio, random_state=self.random_state)

        # Split combined set into validation and test sets.
        texts_val, texts_test, labels_val, labels_test = train_test_split(
            texts_val_test, labels_val_test, test_size=self.test_ratio/(self.test_ratio+self.val_ratio), random_state=self.random_state)

        # Create TextDataset objects for training, validation, and test sets.
        train_dataset = TextDataset(texts_train, labels_train, self.vocab)
        val_dataset = TextDataset(texts_val, labels_val, self.vocab)
        test_dataset = TextDataset(texts_test, labels_test, self.vocab)

        # Create DataLoaders for each split.
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            collate_fn=lambda batch: pad_collate_fn(batch, self.vocab['<pad>'])
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            collate_fn=lambda batch: pad_collate_fn(batch, self.vocab['<pad>'])
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size_test,
            shuffle=False,
            collate_fn=lambda batch: pad_collate_fn(batch, self.vocab['<pad>'])
        )

    def get_loaders(self):
        # Return the training, validation, and test dataloaders.
        return self.train_loader, self.val_loader, self.test_loader

# Example usage
if __name__ == '__main__':
    csv_file = '/home/aicv/work/AI_Human.csv'
    # Initialize DataLoaderFactory with the CSV file path and split ratios.
    data_factory = DataLoaderFactory(csv_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_loader, val_loader, test_loader = data_factory.get_loaders()
    
    # Print number of batches for each loader as a check.
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Validation loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

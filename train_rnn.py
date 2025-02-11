import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
def seed_all(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_all()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Sample Data Loading Function
def load_data():
    questions = ["What is AI?", "Who won the World Cup?", "Where is Mount Everest?", "How does a car engine work?"]
    labels = ["DESC", "HUM", "LOC", "DESC"]
    label_map = {"DESC": 0, "HUM": 1, "LOC": 2}
    encoded_labels = [label_map[label] for label in labels]
    return questions, encoded_labels

# Data Preprocessing
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, questions, labels):
        self.questions = questions
        self.labels = labels
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        encoded = tokenizer(self.questions[idx], padding="max_length", truncation=True, return_tensors="pt")
        return encoded['input_ids'].squeeze(), self.labels[idx]

# Load dataset
questions, labels = load_data()
dataset = TextDataset(questions, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define Simple RNN Model
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=64, output_size=3):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        return self.fc(h_n.squeeze(0))

# Initialize model
vocab_size = tokenizer.vocab_size
model = RNNClassifier(vocab_size).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train_model(num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), torch.tensor(labels).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {correct/total:.4f}")

# Train the model
if __name__ == "__main__":
    train_model()

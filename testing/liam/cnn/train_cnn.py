# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from cnn import Music1DCNN

# Hyperparameters
input_length = 44100  # Example: 1 second of audio at 44.1 kHz sampling rate
num_classes = 10  # Example: number of target classes for classification
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Define the dataset (dummy data for example)
train_loader = [(torch.randn(batch_size, 1, input_length), torch.randint(0, num_classes, (batch_size,))) for _ in range(100)]
test_loader = [(torch.randn(batch_size, 1, input_length), torch.randint(0, num_classes, (batch_size,))) for _ in range(10)]

# Initialize the model
model = Music1DCNN(input_length=input_length, num_classes=num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use MSELoss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_model():
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  # Set the model to training mode

        for i, data in enumerate(train_loader):
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print loss statistics
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    print('Training complete')

# Testing loop
def test_model():
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)

            # For classification tasks
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')

# Main function to run training and testing
if __name__ == '__main__':
    train_model()  # Train the model
    test_model()   # Test the model

import torch
import torch.nn as nn
import torch.nn.functional as F

class SongPopularityClassifier(nn.Module):
    def __init__(self):
        super(SongPopularityClassifier, self).__init__()
        self.fc1 = nn.Linear(6, 10)  
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def train_model(self, data_loader, number_of_epochs, learning_rate):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(number_of_epochs):
            for inputs, labels in data_loader:
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{number_of_epochs}], Loss: {loss.item():.4f}')

    def get_predictions(self, input_data):

        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32)

        self.eval()

        with torch.no_grad():
            logits = self(input_data)

            probabilities = torch.softmax(logits, dim=1)

            predicted_indices = torch.argmax(probabilities, dim=1)

            predicted_classes = [index.item() for index in predicted_indices]

        return probabilities, predicted_classes

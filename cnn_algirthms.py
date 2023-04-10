import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


import wfdb
import numpy as np

from torch.utils.data import Dataset

# path to MIT-BIH dataset
data_path = 'muhammadaarif/IOT/path_mitdb/mit_dataset.csv'

# sampling frequency and duration of each ECG segment
fs = 360
segment_len = 10 # sec

# Load ECG signals and their  annotations
records = ['100', '101', '102'] 
signals = []
labels = []
for record in records:
    signal, fields = wfdb.rdsamp(data_path + record)
    ann = wfdb.rdann(data_path + record, 'atr')
    labels.extend(ann.symbol)
    for i in range(0, len(ann.sample), fs*segment_len):
        start_idx = ann.sample[i]
        end_idx = start_idx + fs*segment_len
        signals.append(signal[start_idx:end_idx, 0])
signals = np.array(signals)
labels = np.array(labels)


train_dataset = ECGRhythmDataset(train_signals, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class ValidationDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ecg = self.data[idx]["ecg"]
        label = self.data[idx]["label"]
        return ecg, label

val_dataset = ValidationDataset(data_path='muhammadaarif/IOT/path_validation_data.csv')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


class ECGRhythmDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        return signal, label


# Define the Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.swish1 = Swish()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.swish2 = Swish()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.swish3 = Swish()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.swish4 = Swish()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.swish1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.swish2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.swish3(x)
        x = self.pool3(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.swish4(x)
        x = self.fc2(x)
        return x

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.lstm = nn.LSTM(input_size=64 * 22, hidden_size=100, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(100, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = Swish()(x)
        x = self.conv2(x)
        x = Swish()(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 22)
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout2(x)
        x = self.fc1(x)
        return x

class Attention(nn.Module):
    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features),
            nn.Tanh(),
            nn.Linear(in_features=in_features, out_features=1)
        )

    def forward(self, x):
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        x = torch.mul(x, weights)
        x = torch.sum(x, dim=1)
        return x
class CNN_LSTM_Attention(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM_Attention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.lstm = nn.LSTM(64, 64, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_att = nn.Linear(64 * 2, 1)
        self.softmax_att = nn.Softmax(dim=1)
        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        att_weights = self.fc_att(x)
        att_weights = self.softmax_att(att_weights)
        x = torch.sum(x * att_weights, dim=1)
        x = self.dropout2(x)
        x = self.fc(x)
        return x



model = CNN(num_classes=num_classes)
#model = CNN_LSTM(num_classes=num_classes)
#model = CNN_LSTM_Attention(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
)

# Train the model
trainer.train(num_epochs=num_epochs)

# Test the model
y_true, y_pred = trainer.test(test_loader)

# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, classes=classes)

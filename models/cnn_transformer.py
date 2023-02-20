import torch
import torch.nn as nn


class CNN_Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super().__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(input_dim[0], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                activation='gelu'
            ),
            num_layers=2
        )

        # fully connected layer
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)

        # reshape
        batch_size, C, H, W = x.shape
        x = x.permute(2, 0, 1, 3).contiguous().view(H, batch_size, C * W)
        x = x.permute(1, 0, 2)

        # transformer
        x = self.transformer(x)

        # average pooling over time
        x = torch.mean(x, dim=1)

        # dropout and fully connected layer
        x = self.dropout(x)
        x = self.fc(x)
        return x

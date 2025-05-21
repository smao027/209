import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_decoder_layers
        )
        self.fc = nn.Linear(input_dim, 2)  # Fully connected layer for binary classification (diabetes: 0 or 1)
        self.softmax = nn.Softmax(dim=-1)  # Softmax for probability distribution

    def forward(self, src, tgt):
        """
        Forward pass of the model.
        :param src: Source sequence tensor of shape (seq_len, batch_size, input_dim)
        :param tgt: Target sequence tensor of shape (seq_len, batch_size, input_dim)
        :return: Probability distribution tensor of shape (batch_size, 2)
        """
        # Pass through the encoder
        memory = self.encoder(src)

        # Pass through the decoder
        output = self.decoder(tgt, memory)

        # Use the last time step's output for classification
        output = self.fc(output[-1])  # Shape: (batch_size, 2)

        # Apply softmax to get probabilities
        probabilities = self.softmax(output)
        return probabilities

class logistic_regression(nn.Module):
    def __init__(self, input_dim):
        super(logistic_regression, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Fully connected layer for binary classification (diabetes: 0 or 1)
        self.softmax = nn.Softmax(dim=-1)  # Softmax for probability distribution

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Probability distribution tensor of shape (batch_size, 2)
        """
        output = self.fc(x)
        probabilities = self.softmax(output)
        return probabilities
    def predict(self, x):
        """
        Predicts the output given the input tensor.
        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Predicted probability tensor of shape (batch_size, 2)
        """
        return self.forward(x)
    
class svm(nn.Module):
    def __init__(self, input_dim):
        super(svm, self).__init__()
        self.fc = nn.Linear(input_dim, 2)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Probability distribution tensor of shape (batch_size, 2)
        """
        output = self.fc(x)
        probabilities = self.softmax(output)
        return probabilities
    def predict(self, x):
        """
        Predicts the output given the input tensor.
        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Predicted probability tensor of shape (batch_size, 2)
        """
        return self.forward(x)
class random_forest(nn.Module):
    def __init__(self, input_dim):
        super(random_forest, self).__init__()
        self.fc = nn.Linear(input_dim, 2)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Probability distribution tensor of shape (batch_size, 2)
        """
        output = self.fc(x)
        probabilities = self.softmax(output)
        return probabilities
    def predict(self, x):
        """
        Predicts the output given the input tensor.
        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Predicted probability tensor of shape (batch_size, 2)
        """
        return self.forward(x)

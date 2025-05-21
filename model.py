import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_encoder_layers
        )
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model=output_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_decoder_layers
        )
        self.fc = nn.Linear(output_dim, 2)  # Fully connected layer for binary classification (diabetes: 0 or 1)
        self.softmax = nn.Softmax(dim=-1)  # Softmax for probability distribution

    def forward(self, src, tgt):
        """
        Forward pass of the model.
        :param src: Source sequence tensor of shape (seq_len, batch_size, input_dim)
        :param tgt: Target sequence tensor of shape (seq_len, batch_size, output_dim)
        :return: Probability distribution tensor of shape (batch_size, 2)
        """
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.fc(output[-1])  # Use the last time step's output for classification
        probabilities = self.softmax(output)
        return probabilities

    def encode(self, src):
        """
        Encodes the source sequence.
        :param src: Source sequence tensor of shape (seq_len, batch_size, input_dim)
        :return: Encoded memory tensor of shape (seq_len, batch_size, input_dim)
        """
        return self.encoder(src)

    def decode(self, tgt, memory):
        """
        Decodes the target sequence using the encoded memory.
        :param tgt: Target sequence tensor of shape (seq_len, batch_size, output_dim)
        :param memory: Encoded memory tensor of shape (seq_len, batch_size, input_dim)
        :return: Decoded output tensor of shape (seq_len, batch_size, output_dim)
        """
        return self.decoder(tgt, memory)

    def predict(self, src, tgt):
        """
        Predicts the output sequence given the source and target sequences.
        :param src: Source sequence tensor of shape (seq_len, batch_size, input_dim)
        :param tgt: Target sequence tensor of shape (seq_len, batch_size, output_dim)
        :return: Predicted probability tensor of shape (batch_size, 2)
        """
        return self.forward(src, tgt)

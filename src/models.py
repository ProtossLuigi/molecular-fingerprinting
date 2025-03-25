import math
from typing import List
from torch import nn
import torch


class MLPModel(nn.Module):
    def __init__(self, hidden_sizes: List[int], activation_layer: type[nn.Module] = nn.ReLU, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.net = nn.Sequential(
            nn.Flatten()
        )
        for in_dim, out_dim in zip([3360] + hidden_sizes, hidden_sizes):
            self.net.append(nn.Linear(in_dim, out_dim))
            self.net.append(activation_layer())
        self.net.append(nn.Linear(hidden_sizes[-1], 1))
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, labels=None):
        # print(inputs, labels)
        inputs = inputs.to(self._dtype())
        logits = self.net(inputs).squeeze()
        if labels is not None:
            labels = labels.to(logits.dtype)
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    def _dtype(self):
        return self.net[1].weight.dtype


class LSTMModel(nn.Module):
    def __init__(self, hidden_size: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.net = nn.LSTM(4, hidden_size, num_layers=3, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, labels=None):
        # print(inputs, labels)
        inputs = inputs.to(self._dtype())
        inputs, _ = self.net(inputs)
        logits = self.classifier(inputs[:, -1, :]).squeeze()
        if labels is not None:
            labels = labels.to(logits.dtype)
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    def _dtype(self):
        return self.classifier.weight.dtype


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30):
        """
        Args
            d_model: Hidden dimensionality of the input.
            max_len: Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerModel(nn.Module):
    def __init__(self, d_model: int = 128, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.embedding_layer = nn.Linear(4, d_model)
        self.pe = PositionalEncoding(d_model, 840)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Linear(d_model, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, labels=None):
        # print(inputs, labels)
        inputs = inputs.to(self._dtype())
        inputs = self.embedding_layer(inputs)
        inputs = self.pe(inputs)
        inputs = self.transformer(inputs)[:, -1, :]
        logits = self.classifier(inputs).squeeze()
        if labels is not None:
            labels = labels.to(logits.dtype)
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    def _dtype(self):
        return self.transformer.layers[0].linear1.weight.dtype

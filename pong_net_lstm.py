#!/usr/bin/env python3

import torch
from torch.autograd import Variable
import torch.nn as nn

# Long Short Term Memory model


class Net(nn.Module):
    def __init__(
        self,
        num_classes,
        imgw,
        imgh,
        hidden_size,
        num_layers,
        dropout,
        numimgseq,
        forcecpu=False,
    ):
        super(Net, self).__init__()

        # Number of eventual output nodes
        self.num_classes = num_classes

        # Number of LSTM layers
        self.num_layers = num_layers

        # Image width and height seperated if needed in the future (e.g., convolve)
        self.imgw = int(imgw)
        self.imgh = int(imgh)
        self.input_size = self.imgw * self.imgh

        # Size of LSTM hidden layer and first fully connected layer
        self.hidden_size = hidden_size

        # dropout percentage (use 0 to turn off)
        self.dropout = dropout

        # Add nodes for multiple layers
        self.hidden_size = self.hidden_size * self.num_layers

        # Number of images per sequence (current + N previous)
        self.numimgseq = numimgseq

        # Override using the GPU if requested
        self.forcecpu = forcecpu

        # Set the processing device (GPU or CPU)
        self.dev = self.getdev()

        # Batch first means the input will be (batches, sequences, features / data)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        ).to(self.dev)

        # Once processed by the LSTM(s) - output is fed into fully connected layer(s)
        fc1_layer_size = self.hidden_size
        fcO_layer_size = int(self.hidden_size * 0.33)

        # First fully connected layer
        self.fc1 = nn.Linear(fc1_layer_size, fcO_layer_size).to(self.dev)

        # Last fully connected layer (O == output)
        self.fcO = nn.Linear(fcO_layer_size, num_classes).to(self.dev)

        self.relu = nn.ReLU().to(self.dev)

    def getdev(self):
        if self.forcecpu == True:
            return "cpu"

        if torch.cuda.is_available():
            # Change from cuda:0 to cpu as needed
            return "cuda:0"
        else:
            return "cpu"

    def forward(self, x):
        batch_size = x.size(0)

        # All sequence lengths should match numimgseq
        x = x.reshape(batch_size, self.numimgseq, self.imgw * self.imgh)

        # Cell state size
        cell_feature_size = self.hidden_size

        # Hidden state
        h_0 = Variable(
            torch.zeros(self.num_layers, batch_size, cell_feature_size).to(self.dev)
        )

        # Internal state
        c_0 = Variable(
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.dev)
        )

        # Propagate input through LSTM
        _, (hn, _) = self.lstm(x, (h_0, c_0))

        # Get the last hidden state
        out = hn.view(-1, cell_feature_size)

        # Time to reluuuuuuuuuu
        out = self.relu(out)

        # LSTM output into fully connected layer(s)
        out = self.fc1(out)
        out = self.relu(out)

        out = self.fcO(out)
        out = torch.sigmoid(out)

        return out


if __name__ == "__main__":
    pass

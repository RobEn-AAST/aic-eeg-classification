import torch
import torch.nn as nn
from .convolution import DepthWiseConv2D, SeperableConv2D
from .lstm import LSTMModel


class SSVEPClassifier(nn.Module):
    # EEG Net Based
    # todo look at this https://paperswithcode.com/paper/a-transformer-based-deep-neural-network-model
    def __init__(self, n_electrodes=16, n_samples=128, out_dim=4, dropout=0.25, kernLength=256, F1=96, D=1, F2=96, hidden_dim=100, layer_dim=1):
        super().__init__()

        # B x C x T
        self.block_1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False),
            nn.BatchNorm2d(F1),
            #
            DepthWiseConv2D(F1, (n_electrodes, 1), dim_mult=D, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),  # todo try making this max pool
            nn.Dropout(dropout),
            #
            SeperableConv2D(F1 * D, F2, kernel_size=(1, 16), padding="same", bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.lstm_head = LSTMModel(F2, hidden_dim, layer_dim, out_dim)

    def forward(self, x: torch.Tensor):
        """expected input shape: BxCxT"""
        x = x.unsqueeze(1)
        y = self.block_1(x)  # B x F1 x 1 x time_sub

        y = y.squeeze(2)  # B x F1 x time_sub
        y = y.permute(0, 2, 1)  # B x time_sub x F1
        y = self.lstm_head(y)

        return y


if __name__ == "__main__":
    model = SSVEPClassifier()
    print(model)
    x = torch.randn(32, 16, 128)  # Batch size of 32, 16 electrodes, 128 time points
    output = model(x)
    print(output.shape)  # Should be (32, out_dim)

import torch
import torch.nn as nn


class RDB(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 32, k_size: int = 3):
        """
        Initializes a Residual Dense Block (RDB).

        Args:
            in_channels (int): The number of input channels to this RDB.
            growth_rate (int): The number of output channels for each internal convolutional layer
            k_size (int): The 'k' for the asymmetric kx1 and 1xk convolution kernels.
                          The sources indicate that this decomposition is effective for reducing
                          computation when `k_size` is greater than 2 [4], suggesting typical values
                          like 3, 5, or 7.
        """
        super(RDB, self).__init__()

        # --- Internal Layers with Dense Connections ---

        # Layer 1: First 1x1 Convolution (Conv_1x1_A) [Figure 4, 18]
        self.conv1_1x1_A = nn.Conv2d(in_channels, growth_rate, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        # Layer 2: Asymmetric kx1 Convolution (Conv_kx1) [Figure 4, 18]
        self.conv2_kx1 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=(k_size, 1), stride=1, padding=(k_size // 2, 0))
        self.relu2 = nn.ReLU(inplace=True)

        # Layer 3: Asymmetric 1xk Convolution (Conv_1xk) [Figure 4, 18]
        self.conv3_1xk = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, kernel_size=(1, k_size), stride=1, padding=(0, k_size // 2))
        self.relu3 = nn.ReLU(inplace=True)  # Inferred ReLU [Conversation History, Figure 6].

        # Batch Normalization (BN) Layer
        self.bn_layer = nn.BatchNorm2d(in_channels + 3 * growth_rate)
        self.relu_bn = nn.ReLU(inplace=True)  # Inferred ReLU after BN, consistent with Figure 6 [Conversation History].

        # Final Internal Layer: Second 1x1 Convolution (Conv_1x1_B)
        self.conv4_1x1_B = nn.Conv2d(in_channels + 3 * growth_rate, in_channels, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x_input_rdb: torch.Tensor) -> torch.Tensor:
        accumulated_dense_features = [x_input_rdb]

        # Helper function for concatenation along the channel dimension (dim=1 for NCHW format).
        # This is the "dense connection" mechanism, feeding all prior outputs into subsequent layers [14, Figure 4].
        def _concatenate_features(features_list):
            return torch.cat(features_list, dim=1)

        # --- Layer 1: Conv_1x1_A ---
        f1_out = self.conv1_1x1_A(_concatenate_features(accumulated_dense_features))
        f1_out = self.relu1(f1_out)
        accumulated_dense_features.append(f1_out)

        # --- Layer 2: Conv_kx1 ---
        f2_out = self.conv2_kx1(_concatenate_features(accumulated_dense_features))
        f2_out = self.relu2(f2_out)
        accumulated_dense_features.append(f2_out)

        # --- Layer 3: Conv_1xk ---
        f3_out = self.conv3_1xk(_concatenate_features(accumulated_dense_features))
        f3_out = self.relu3(f3_out)
        accumulated_dense_features.append(f3_out)

        # --- Batch Normalization (BN) Layer ---
        bn_out = self.bn_layer(_concatenate_features(accumulated_dense_features))
        bn_out = self.relu_bn(bn_out)

        # --- Final Internal Layer: Conv_1x1_B ---
        f_transformed_block_output = self.conv4_1x1_B(bn_out)
        f_transformed_block_output = self.relu4(f_transformed_block_output)

        # --- Final Residual Connection (Addition) ---
        x_output_rdb = f_transformed_block_output + x_input_rdb

        return x_output_rdb


if __name__ == "__main__":
    # Example usage of the RDB class
    rdb = RDB(in_channels=64, growth_rate=32, k_size=3)
    input_tensor = torch.randn(1, 64, 128, 128)  # Batch size of 1, 64 channels, 128x128 spatial dimensions
    output_tensor = rdb(input_tensor)
    print(output_tensor.shape)  # Should be (1, 64, 128, 128)
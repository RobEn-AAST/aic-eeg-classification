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


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """
        Args:
            in_channels (int): Number of input channels to the attention module.
            reduction_ratio (int): The reduction ratio for the MultiLayer Perceptron (MLP)
        """
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)

        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        channel_attention = self.sigmoid(avg_out + max_out)

        return x * channel_attention  # This is the output used by the Parallel CBAM for fusion.


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size (int): The size of the convolution kernel for spatial attention.
                               The sources explicitly mention a 7x7 kernel [3].
        """
        super(SpatialAttentionModule, self).__init__()
        # Padding to maintain spatial dimensions for the given kernel_size.
        # For a 7x7 kernel, padding = (7-1)/2 = 3.
        assert kernel_size in (3, 5, 7), "kernel size must be 3, 5, or 7"  # General practice for common kernels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.conv(spatial_input)

        spatial_attention = self.sigmoid(spatial_attention)
        return x * spatial_attention


class ParallelCBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        """
        Improved Attention Mechanism: Parallel CBAM as described in LFANN

        Args:
            in_channels (int): The number of input channels for the feature map.
            reduction_ratio (int): Reduction ratio for the Channel Attention Module's MLP.
            spatial_kernel_size (int): Kernel size for the Spatial Attention Module's convolution.
        """
        super(ParallelCBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_refined_features = self.channel_attention(x)
        spatial_refined_features = self.spatial_attention(x)

        output = channel_refined_features + spatial_refined_features
        return output


class LAFFN(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 2, base_channels: int = 32):
        """
        Lightweight Feature Fusion Network (LAFFN) based on improved attention mechanism (Parallel CBAM)
        and Residual Dense Blocks (RDB)
        O
        Args:
            in_channels (int): Number of input channels for the 2D-TFI.
            num_classes (int): Number of output classes for MI EEG classification.
            base_channels (int): Base number of channels after the initial convolutional layer.
                                 This value scales up in subsequent RDB blocks. Recommended: 32.
        """
        super(LAFFN, self).__init__()
        self.initial_conv = nn.Sequential(nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True))


        # Block 1: RDB1 -> Parallel CBAM -> MaxPool
        # Channels: base_channels (32) -> base_channels * 2 (64)
        # Spatial: 80x80 -> 40x40 (via MaxPool)
        self.block1 = nn.Sequential(RDB(base_channels, base_channels * 2), ParallelCBAM(base_channels * 2), nn.MaxPool2d(kernel_size=2, stride=2))  # RDB1 increases channels

        # Block 2: RDB2 -> Parallel CBAM -> MaxPool
        # Channels: base_channels * 2 (64) -> base_channels * 4 (128)
        # Spatial: 40x40 -> 20x20 (via MaxPool)
        self.block2 = nn.Sequential(RDB(base_channels * 2, base_channels * 4), ParallelCBAM(base_channels * 4), nn.MaxPool2d(kernel_size=2, stride=2))  # RDB2 increases channels

        # Block 3: RDB3 -> Parallel CBAM -> MaxPool
        # Channels: base_channels * 4 (128) -> base_channels * 4 (128) - maintains for final pooling
        # Spatial: 20x20 -> 10x10 (via MaxPool)
        self.block3 = nn.Sequential(RDB(base_channels * 4, base_channels * 4), ParallelCBAM(base_channels * 4), nn.MaxPool2d(kernel_size=2, stride=2))  # RDB3 maintains channels

        # Final MaxPool to get to 5x5 spatial dimensions
        # Spatial: 10x10 -> 5x5 (via MaxPool)
        self.final_spatial_reduction = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling Layer [14, Figure 5]
        # Input to this layer will be (Batch_size, 128, 5, 5)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Classification Layer (Fully Connected Layer) [14, Figure 5]
        # Takes the 128 features from global pooling and outputs to num_classes
        self.classifier = nn.Linear(base_channels * 4, num_classes)  # 128 channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)  # Output: (N, 32, 80, 80)

        x = self.block1(x)  # Output: (N, 64, 40, 40)
        x = self.block2(x)  # Output: (N, 128, 20, 20)
        x = self.block3(x)  # Output: (N, 128, 10, 10)

        x = self.final_spatial_reduction(x)  # Output: (N, 128, 5, 5)

        x = self.global_avg_pool(x)  # Output: (N, 128, 1, 1)
        x = torch.flatten(x, 1)  # Output: (N, 128)

        x = self.classifier(x)  # Output: (N, num_classes)
        return x


if __name__ == "__main__":
    lfann = LAFFN(in_channels=4, num_classes=2, base_channels=32)
    print(lfann)
    # Test with a dummy input tensor of shape (batch_size, channels, height, width)
    dummy_input = torch.randn(8, 4, 160, 160)
    output = lfann(dummy_input)
    print("Output shape:", output.shape)  # Should be (8, 2)
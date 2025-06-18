did the extra data improve or hurt the model?

is doing multi target network better than separate networks for each target?

is transfering data to frequency domain better than time domain?


# Resources used:
https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py for tensorflow implementation EEGNet SSVEP
https://github.com/amrzhd/EEGNet
https://ieeexplore.ieee.org/document/9328251 (for explanation on EEGNet for SSVEP)
https://onlinelibrary.wiley.com/doi/10.1002/eng2.12827 (tricks for eegnet with lstm)

lstm params: [I 2025-06-18 04:01:38,445] Trial 12 finished with value: 0.65625 and parameters: {'n_electrodes': 32, 'n_samples': 256, 'dropout': 0.33066508963955576, 'kernLength': 256, 'F1': 128, 'D': 2, 'F2': 96, 'hidden_dim': 256, 'layer_dim': 3, 'window_length': 160, 'stride': 3, 'lr': 0.00030241790493218325, 'batch_size': 64}. Best is trial 12 with value: 0.65625.
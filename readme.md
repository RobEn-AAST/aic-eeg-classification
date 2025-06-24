did the extra data improve or hurt the model?

is doing multi target network better than separate networks for each target?

is transfering data to frequency domain better than time domain?


# Resources used:
https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py for tensorflow implementation EEGNet SSVEP
https://github.com/amrzhd/EEGNet
https://ieeexplore.ieee.org/document/9328251 (for explanation on EEGNet for SSVEP)
https://onlinelibrary.wiley.com/doi/10.1002/eng2.12827 (tricks for eegnet with lstm with preprocessing tehcniques)

# To look at
https://arxiv.org/pdf/2403.03276v2 (ARNN: Attentive Recurrent Neural Network for Multi-channel EEG Signals to Identify
Epileptic Seizures)
https://paperswithcode.com/paper/a-transformer-based-deep-neural-network-model (transformer based ssvep classification)


ssvep_PO8_OZ_PZ params: 50 epoch test tuner
channels: eeg_channels = ["PO8", "OZ", "PZ"]
[I 2025-06-23 23:33:32,774] Trial 17 finished with value: 0.7183263207106124 and parameters: {'window_length': 256, 'batch_size': 64, 'kernLength': 256, 'F1': 32, 'D': 3, 'F2': 32, 'hidden_dim': 256, 'layer_dim': 3, 'dropout': 0.26211635308091535, 'lr': 0.0003746351873334935}. Best is trial 17 with value: 0.7183263207106124.
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


ssvep_honored_one params:
channels: eeg_channels = ["PO8", "OZ"]
[I 2025-06-23 21:39:53,896] Trial 36 finished with value: 0.5896541702993315 and parameters: {'window_length': 256, 'batch_size': 64, 'kernLength': 256, 'F1': 16, 'D': 3, 'F2': 64, 'hidden_dim': 64, 'layer_dim': 1, 'dropout': 0.10328170267309397, 'lr': 0.0018182233882257615}. Best is trial 36 with value: 0.5896541702993315.

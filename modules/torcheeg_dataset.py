# https://commons.wikimedia.org/wiki/File:International_10-20_system_for_EEG-MCN.svg
# https://torcheeg.readthedocs.io/en/latest/generated/torcheeg.datasets.BCICIV2aDataset.html
from torcheeg.datasets import BCICIV2aDataset
from torcheeg import transforms

dataset = BCICIV2aDataset(root_path='./BCICIV_2a_mat',
                          online_transform=transforms.Compose([
                              transforms.To2d(),
                              transforms.ToTensor()
                          ]),
                          label_transform=transforms.Compose([
                              transforms.Select('label'),
                              transforms.Lambda(lambda x: x - 1)
                          ]))
print(dataset[0])
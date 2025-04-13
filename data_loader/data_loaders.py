from torchvision import transforms
from base import BaseDataLoader
from data_loader.dataset import SCM_Dataset


class SCMDataLoader(BaseDataLoader):
    """
    SCM data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = SCM_Dataset()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

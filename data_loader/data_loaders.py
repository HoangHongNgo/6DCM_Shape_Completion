from torchvision import transforms
from base.base_data_loader import BaseDataLoader
from data_loader.dataset import SCM_Dataset
from torch.utils.data import DataLoader, RandomSampler


class SCMDataLoader(BaseDataLoader):
    """
    SCM data loading using BaseDataLoader
    """

    def __init__(self, data_dir, camera, train_batch_size, val_batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # Setup the dataset
        train_data = SCM_Dataset(
            gn_root=data_dir, camera=camera, split='train', pred_depth=True)
        val_data = SCM_Dataset(
            gn_root=data_dir, camera=camera, split='val', pred_depth=True
        )

        # Setup sample set
        train_sample = RandomSampler(
            train_data, replacement=True, num_samples=int(len(train_data)/2))

        # Setup dataloader
        train_loader = DataLoader(
            train_data, sampler=train_sample, batch_size=train_batch_size, num_workers=train_batch_size)  # TODO change num worker
        val_loader = DataLoader(
            val_data, batch_size=val_batch_size, num_workers=val_batch_size)

        super().__init__(train_loader=train_loader, val_loader=val_loader)

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    This class extends PyTorch's DataLoader to handle dataset splitting
    into training and validation sets and to manage batching.
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        # Initialize data loader with dataset, batch size, shuffle, validation split, and number of workers
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0  # Index to track batches
        self.n_samples = len(dataset)  # Total number of samples in the dataset

        # Split the dataset into training and validation sets based on the validation split
        self.sampler, self.valid_sampler = self._split_sampler(
            self.validation_split)

        # Store the initialization arguments for the DataLoader
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        # Initialize the base DataLoader with training sampler (without shuffle)
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        """
        Splits the dataset into training and validation sets based on the split ratio or size.
        Returns two samplers: one for training and one for validation.
        """

        # If the split is 0, return no validation data
        if split == 0.0:
            return None, None

        # Generate an array of indices representing the dataset samples
        idx_full = np.arange(self.n_samples)

        # Shuffle the indices for randomness
        np.random.seed(0)  # Fixed seed for reproducibility
        np.random.shuffle(idx_full)

        # If split is an integer, it represents the number of validation samples
        # Otherwise, it is a fraction representing the percentage of the dataset for validation
        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            # Calculate the number of validation samples based on the split percentage
            len_valid = int(self.n_samples * split)

        # Split the indices into validation and training indices
        # First 'len_valid' samples are for validation
        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(
            0, len_valid))  # The rest are for training

        # Create random samplers for training and validation sets
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # Disable shuffling for the train sampler since it's handled by the sampler
        self.shuffle = False
        # Update the number of training samples
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        """
        Returns a DataLoader for the validation set using the validation sampler.
        If no validation set is defined, returns None.
        """
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

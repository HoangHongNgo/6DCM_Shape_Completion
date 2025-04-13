import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):  # TODO can start writing train now!
    """
    Trainer class
    This class extends the BaseTrainer and provides the training and validation logic
    for the model. It also handles metric tracking, logging, and learning rate scheduling.
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        """
        Initialize the trainer with model, optimizer, data loaders, and configuration

        :param model: The model to be trained
        :param criterion: The loss function
        :param metric_ftns: List of metrics to evaluate the model
        :param optimizer: The optimizer used for training
        :param config: Configuration dictionary containing training parameters
        :param device: The device (CPU or GPU) where the model and data will reside
        :param data_loader: Training data loader
        :param valid_data_loader: Optional validation data loader
        :param lr_scheduler: Optional learning rate scheduler
        :param len_epoch: Optional number of iterations per epoch
        """
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader

        # If len_epoch is None, use epoch-based training
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            # If len_epoch is provided, use iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        # Optional validation data loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        # Optional learning rate scheduler
        self.lr_scheduler = lr_scheduler

        # Log step frequency for visualization (e.g., logging every few steps)
        self.log_step = int(np.sqrt(data_loader.batch_size))

        # Initialize metric trackers for training and validation
        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.
        This method performs one epoch of training, including computing loss, updating metrics, 
        logging, and visualizing the inputs.

        :param epoch: Integer, current training epoch.
        :return: A dictionary containing average loss and metrics for the epoch.
        """
        self.model.train()  # Set the model to training mode
        self.train_metrics.reset()  # Reset the training metrics tracker

        for batch_idx, (data, target) in enumerate(self.data_loader):
            # Move data and target to the configured device (CPU or GPU)
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()  # Clear previous gradients
            output = self.model(data)  # Forward pass through the model
            loss = self.criterion(output, target)  # Compute the loss
            loss.backward()  # Backward pass (compute gradients)
            self.optimizer.step()  # Update model weights

            # Log the step (for Tensorboard visualization)
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            # Update training metrics
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            # Log every few steps (based on log_step)
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),  # Print progress
                    loss.item()))
                self.writer.add_image('input', make_grid(
                    data.cpu(), nrow=8, normalize=True))  # Log images

            if batch_idx == self.len_epoch:  # Stop after one full epoch
                break

        log = self.train_metrics.result()  # Get the training metrics

        # If validation data loader is available, run validation
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            # Combine training and validation logs
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        # Step the learning rate scheduler if provided
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log  # Return the training log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch.
        This method performs one epoch of validation, computes the loss and metrics,
        and logs images and parameter histograms.

        :param epoch: Integer, current training epoch.
        :return: A dictionary containing validation loss and metrics.
        """
        self.model.eval()  # Set the model to evaluation mode
        self.valid_metrics.reset()  # Reset the validation metrics tracker

        with torch.no_grad():  # Disable gradient computation during validation
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)  # Forward pass
                loss = self.criterion(output, target)  # Compute loss

                # Log validation steps for Tensorboard
                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(
                    data.cpu(), nrow=8, normalize=True))  # Log validation images

        # Add histograms of model parameters to Tensorboard for visualization
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return self.valid_metrics.result()  # Return validation metrics

    def _progress(self, batch_idx):
        """
        Generate progress string for batch-based training

        :param batch_idx: Current batch index
        :return: A string representing the current progress
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

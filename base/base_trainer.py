import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    This class provides the structure for training models, saving checkpoints, 
    handling early stopping, and monitoring performance during training.
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        # Initialize the trainer with necessary components like model, criterion, optimizer, etc.
        self.config = config
        self.logger = config.get_logger(
            'trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        # Extract training configuration from the provided config
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # Configuration for model performance monitoring and saving best models
        if self.monitor == 'off':
            self.mnt_mode = 'off'  # No performance monitoring
            self.mnt_best = 0  # No best metric to track
        else:
            # Monitor based on min or max metric
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            # Initialize the best performance metric value
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get(
                'early_stop', inf)  # Early stopping criteria
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1  # Start training from epoch 1

        self.checkpoint_dir = config.save_dir  # Directory to save model checkpoints

        # Initialize Tensorboard writer for visualizing training progress
        self.writer = TensorboardWriter(
            config.log_dir, self.logger, cfg_trainer['tensorboard'])

        # Resume from checkpoint if specified
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for a single epoch.

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic that loops through all epochs.
        It trains the model for the specified number of epochs, 
        tracks performance, and saves checkpoints.
        """
        not_improved_count = 0  # Counter for early stopping if no improvement
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Call the abstract method to train for this epoch
            result = self._train_epoch(epoch)

            # Log the results of the epoch
            log = {'epoch': epoch}
            log.update(result)

            # Print the logged information to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            best = False  # Flag to track if the model improved during this epoch
            if self.mnt_mode != 'off':  # If monitoring is enabled
                try:
                    # Check if the model's performance improved according to the monitored metric
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode ==
                                'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    # If the monitored metric is not found, disable performance monitoring
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                # If the model improved, save the current best performance and reset the counter
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                # If no improvement for a set number of epochs, stop training early
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn't improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            # Save checkpoint every 'save_period' epochs
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Save the model checkpoint to disk

        :param epoch: Current epoch number
        :param save_best: Whether to save the best model checkpoint
        """
        # Prepare the state dictionary with model architecture, epoch, optimizer, and other config info
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir /
                       'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)  # Save the checkpoint file
        self.logger.info("Saving checkpoint: {} ...".format(filename))

        # If this is the best model, save it with a 'model_best.pth' name
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume training from a saved checkpoint

        :param resume_path: Path to the checkpoint file
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + \
            1  # Resume from the next epoch
        # Load the best performance metric
        self.mnt_best = checkpoint['monitor_best']

        # Ensure architecture and optimizer match before resuming
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning(
                "Warning: Architecture configuration in config differs from checkpoint.")
        self.model.load_state_dict(
            checkpoint['state_dict'])  # Load model weights

        # Load optimizer state only if optimizer type matches
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning(
                "Warning: Optimizer type differs from checkpoint. Optimizer parameters not resumed.")
        else:
            self.optimizer.load_state_dict(
                checkpoint['optimizer'])  # Load optimizer state

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    This class serves as a parent class for all model architectures. It defines
    the common structure and methods that should be implemented by any model
    that inherits from it.
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        This method should define the computation performed by the model during the forward pass.

        :return: Model output after processing inputs
        """
        # Since this is an abstract method, it raises NotImplementedError to force subclasses to implement it
        raise NotImplementedError

    def __str__(self):
        """
        Returns a string representation of the model, including the number of trainable parameters.
        This method overrides the default `__str__` to provide additional details about the model.
        """
        # Filter the model parameters to include only those that require gradients (i.e., trainable parameters)
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())

        # Sum up the total number of trainable parameters by multiplying the size of each parameter tensor
        params = sum([np.prod(p.size()) for p in model_parameters])

        # Return the string representation of the model with the additional information about trainable parameters
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

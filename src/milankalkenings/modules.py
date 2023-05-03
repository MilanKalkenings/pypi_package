from typing import TypedDict, Union, List
import torch
from abc import ABC, abstractmethod
import torch.nn as nn

ModuleOutput = TypedDict('ModuleOutput', {'loss': torch.Tensor, 'scores': torch.Tensor}, total=False)


class Module(ABC, nn.Module):
    """
    abstract wrapper for torch.nn.Module.
    adds abstract methods for (un)freezing pretrained layers.
    ensures input and output format for the forward pass.
    """
    def __init__(self):
        """
        Initialize the base module.
        """
        super().__init__()

    def freeze_pretrained(self):
        """
        Freeze the pretrained layers of the module.
        """
        pass

    def unfreeze_pretrained(self):
        """
        Unfreeze the pretrained layers of the module.
        """
        pass

    @abstractmethod
    def forward(self,
                x: Union[torch.Tensor, List[torch.Tensor]],
                y: Union[torch.Tensor, List[torch.Tensor]]) -> ModuleOutput:
        """
        performs forward pass

        :param x: input
        :type x: Union[torch.Tensor, List[torch.Tensor]]

        :param y: supervised labels
        :type y: Union[torch.Tensor, List[torch.Tensor]]

        :return: output of the forward pass
        :rtype: ModuleOutput
        """
        pass


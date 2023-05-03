import time
import random
import numpy as np
import torch

from . modules import Module


def largest_divisor(n: int) -> int:
    """
    :param n: a number
    :type n: int

    :return: largest int divisor of n
    :rtype: int
    """
    i = n // 2
    while i > 1:
        if n % i == 0:
            return i
        i -= 1
    return 1


def save_secure(module: Module, file: str):
    """
    securely saves a module in a file located at "path"

    :param module: Module loaded from the file
    :type module: Module

    :param file: module file
    :type file: str
    """
    while True:
        try:
            torch.save(module, file)
            break
        except:
            time.sleep(0.1)
    return module


def load_secure(file: str) -> Module:
    """
    securely loads a Module from a file located at "path"

    :param file: module file
    :type file: str
    :return: Module loaded from the file
    :rtype: Module
    """
    while True:
        try:
            module = torch.load(file)
            break
        except:
            time.sleep(0.1)
    return module


def make_reproducible(seed: int = 1):
    """
    make train/eval process reproducible.
    :param seed: seed for random number generators
    :type seed: int
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_printoptions(sci_mode=False)
    torch.set_printoptions(threshold=100_000)
    np.set_printoptions(suppress=True)

from abc import ABC, abstractmethod
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import matplotlib.pyplot as plt
import itertools
import time
from typing import TypedDict, Union, List, Tuple

# documentation
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

    @abstractmethod
    def freeze_pretrained(self):
        """
        Freeze the pretrained layers of the module.
        """
        pass

    @abstractmethod
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


class Setup:
    """
    stores all the configurations for train/eval process

    :param loader_train: DataLoader for the training set
    :type loader_train: torch.utils.data.DataLoader

    :param loader_val: DataLoader for the validation set
    :type loader_val: torch.utils.data.DataLoader

    :param loader_test: DataLoader for the test set
    :type loader_test: torch.utils.data.DataLoader

    :param device: device to run the training process on.
    :type device: str

    :param monitor_n_losses: the number of steps after which the loss slope is printed.
    :type monitor_n_losses: int

    :param checkpoint_initial: path to the initial checkpoint.
    :type checkpoint_initial: str

    :param checkpoint_running: path to the running checkpoint.
    :type checkpoint_running: str

    :param checkpoint_final: path to the final checkpoint.
    :type checkpoint_final: str

    :param lrrt_n_batches: number of batches used in the learning rate range test.
    :type lrrt_n_batches: int

    :param lrrt_slope_desired: exclusive border used in the learning rate range test.
    :type lrrt_slope_desired: float

    :param lrrt_max_decays: maximum number of learning rate decays performed in the learning rate range test.
    :type lrrt_max_decays: int

    :param lrrt_decay: decay factor used in the learning rate range test.
    :type lrrt_decay: float

    :param lrrt_initial_candidates: initial candidate learning rates used in the learning rate range test.
    :type lrrt_initial_candidates: np.ndarray

    :param overkill_initial_lr: initial learning rate used in overkill training
    :type overkill_initial_lr: float

    :param overkill_decay: learning rate decay factor used in overkill training
    :type overkill_decay: float

    :param overkill_max_violations: maximum number of violations allowed in overkill training
    :type overkill_max_violations: int

    :param overkill_max_decays: maximum number of learning rate decays allowed in overkill training
    :type overkill_max_decays: int

    :param es_max_violations: maximum number of violations allowed in early stopping
    :type es_max_violations: int

    :param optimizer_class: optimizer class used for the training process
    :type optimizer_class: type[torch.optim.Optimizer]
    """
    def __init__(self,
                 loader_train: DataLoader,
                 loader_val: DataLoader,
                 loader_test: DataLoader,
                 device: str = "cpu",
                 monitor_n_losses: int = 50,
                 checkpoint_initial: str = "../monitoring/checkpoint_initial.pkl",
                 checkpoint_running: str = "../monitoring/checkpoint_running.pkl",
                 checkpoint_final: str = "../monitoring/checkpoint_final.pkl",
                 lrrt_n_batches: int = 49,
                 lrrt_slope_desired: float = 0,
                 lrrt_max_decays: int = 0,
                 lrrt_decay: float = 0.9,
                 lrrt_initial_candidates: np.ndarray = np.array([1e-3, 1e-4, 1e-6]),
                 overkill_initial_lr: float = 0.005,
                 overkill_decay: float = 0.9,
                 overkill_max_violations: int = None,
                 overkill_max_decays: int = 10,
                 es_max_violations: int = 2,
                 optimizer_class=torch.optim.Adam):
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.device = device
        self.optimizer_class = optimizer_class
        self.monitor_n_losses = monitor_n_losses
        self.checkpoint_initial = checkpoint_initial
        self.checkpoint_running = checkpoint_running
        self.checkpoint_final = checkpoint_final

        # lrrt
        self.lrrt_n_batches = lrrt_n_batches
        self.lrrt_slope_desired = lrrt_slope_desired
        self.lrrt_max_decays = lrrt_max_decays
        self.lrrt_decay = lrrt_decay
        self.lrrt_initial_candidates = lrrt_initial_candidates

        # overkill
        self.overkill_initial_lr = overkill_initial_lr
        self.overkill_decay = overkill_decay
        if overkill_max_violations is None:
            self.overkill_max_violations = len(self.loader_train)
        else:
            self.overkill_max_violations = overkill_max_violations
        self.overkill_max_decays = overkill_max_decays

        # early stopping
        self.es_max_violations = es_max_violations


class BatchHandler:
    """
    handles data in batch wise manner for train/eval.

    :ivar setup: configurations
    :vartype attr1: Setup
    """
    def __init__(self, setup: Setup):
        self.setup = setup

    def forward_batch(self,
                      module: Module,
                      batch: Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]) -> ModuleOutput:
        """
        performs a forward pass with a given module and batch.

        :param module: torch module with specified inputs and outputs
        :type module: Module

        :param batch: batch containing x and y
        :type batch: Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]

        :return: output of the forward pass
        :rtype: ModuleOutput
        """

        x, y = batch
        x = x.to(self.setup.device)
        y = y.to(self.setup.device)
        return module(x=x, y=y)

    def train_batch(self,
                    module: Module,
                    optimizer: Optimizer,
                    batch: Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]],
                    freeze_pretrained: bool = False) -> float:
        """
        train on one batch

        :param module: module that has to be trained
        :type module: Module

        :param optimizer: optimizer used to perform the update step
        :type optimizer: Optimizer

        :param batch: batch containing x, y
        :type batch: Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]

        :param freeze_pretrained: determines if pretrained layers are frozen
        :type freeze_pretrained: bool

        :return: float representation of the loss
        :rtype: float
        """
        # freeze/unfreeze here: longer runtime, better encapsulation
        if freeze_pretrained:
            module.freeze_pretrained()
        else:
            module.freeze_pretrained()
        module.train()
        module.zero_grad()
        loss = self.forward_batch(module=module, batch=batch)["loss"]
        loss.backward()
        optimizer.step()
        return float(loss)

    def loss_batch_eval(self,
                        module: Module,
                        batch: Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]) -> float:
        """
        calculates the loss on one batch in evaluation mode

        :param module: module to evaluate
        :type module: Module

        :param batch: batch containing x, y
        :type batch: Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]

        :return: float representation of the batch loss
        :rtype: float
        """
        module.eval()
        with torch.no_grad():
            return float(self.forward_batch(module=module, batch=batch)["loss"])

    def train_n_batches(self,
                        module: Module,
                        optimizer: Optimizer,
                        n_batches: int,
                        loader: DataLoader,
                        freeze_pretrained: bool) -> Tuple[List[float], float, float]:
        """
        trains a given module on n batches of a given data loader.

        :param module: module to be trained
        :type module: Module

        :param optimizer: optimizer to train the module
        :type optimizer: Optimizer

        :param n_batches: number of batches for training
        :type n_batches: int

        :param loader: data loader used to draw the training data from
        :type loader: DataLoader

        :param freeze_pretrained: determines if pretrained layers are frozen
        :type freeze_pretrained: bool

        :return:
        """
        losses = []
        for train_iter, batch in enumerate(loader):
            if train_iter == n_batches:
                break

            losses.append(self.train_batch(module=module,
                                           optimizer=optimizer,
                                           batch=batch,
                                           freeze_pretrained=freeze_pretrained))
            if (len(losses) % self.setup.monitor_n_losses) == 0:
                losses_last = np.array(losses[-self.setup.monitor_n_losses:])
                slope_last, _ = np.polyfit(x=np.arange(len(losses_last)), y=losses_last, deg=1)
                print("iter", train_iter + 1, "mean loss", losses_last.mean(), "loss slope", slope_last)
        slope_total, bias_total = np.polyfit(x=np.arange(len(losses)), y=losses, deg=1)
        return losses, float(slope_total), float(bias_total)

    def lrrt(self,
             loader: DataLoader,
             freeze_pretrained: bool = False) -> Tuple[float, float]:
        """
        Learning Rate Range Test; basic idea:
        for each learning rate in a set of learning rate candidates:
            load a checkpoint
            train from the checkpoint on a small amount of batches
            determine the slope of the batch losses
            return the learning rate that creates the steepest negative slope

        modified to rerun with a decayed set of learning rate candidates
        until a max number of iterations or a certain slope is reached.

        :param loader: data loader to draw batches from
        :type loader: DataLoader

        :param freeze_pretrained: determines if pretrained layers are frozen
        :type freeze_pretrained: bool

        :return: best learning rate, best loss slope
        :rtype: List[float, float]
        """
        print("lr search using lrrt")
        slope_desired_found = False
        candidate_lrs = self.setup.lrrt_initial_candidates
        lr_best_total = np.inf
        slope_best_total = np.inf
        for decay_it in range(self.setup.lrrt_max_decays + 1):
            candidate_slopes = np.zeros(shape=len(candidate_lrs))
            for i, lr_candidate in enumerate(candidate_lrs):
                module = load_secure(self.setup.checkpoint_running)
                optimizer = self.setup.optimizer_class(params=module.parameters(), lr=lr_candidate)
                candidate_slopes[i] = self.train_n_batches(module=module,
                                                           optimizer=optimizer,
                                                           n_batches=self.setup.lrrt_n_batches,
                                                           freeze_pretrained=freeze_pretrained,
                                                           loader=loader)[1]
            best_candidate_slope_id = np.argmin(candidate_slopes)
            best_candidate_slope = candidate_slopes[best_candidate_slope_id]
            best_candidate_lr = candidate_lrs[best_candidate_slope_id]
            if best_candidate_slope < slope_best_total:
                slope_best_total = best_candidate_slope
                lr_best_total = best_candidate_lr
            if slope_best_total < self.setup.lrrt_slope_desired:
                slope_desired_found = True
                break
            else:
                candidate_lrs = candidate_lrs * self.setup.lrrt_decay
        if not slope_desired_found:
            print("lr with desired loss slope", self.setup.lrrt_slope_desired, "not found. using approx best lr")
        print("best loss slope", slope_best_total, "best lr", lr_best_total)
        return lr_best_total, slope_best_total

    def loss_epoch_eval(self, module: Module, loader_eval: DataLoader) -> float:
        """
        calculates the epoch loss in evaluation mode

        :param module: module to be evaluated
        :type module: Module

        :param loader_eval: data loader used for evaluation
        :type loader_eval: DataLoader

        :return: epoch loss
        :rtype: float
        """
        batch_losses = np.zeros(len(loader_eval))
        for batch_nr, batch in enumerate(loader_eval):
            batch_losses[batch_nr] = self.loss_batch_eval(module=module, batch=batch)
        return float(batch_losses.mean())

    def losses_epoch_eval(self, module: Module) -> Tuple[float, float]:
        """
        wrapper for loss_epoch_eval
        calculates epoch loss for training data and validation data

        :param module: module to be evaluated
        :type module: Module

        :return: training epoch loss, validation epoch loss
        :rtype: List[float, float]
        """
        loss_epoch_train = self.loss_epoch_eval(module=module, loader_eval=self.setup.loader_train)
        loss_epoch_val = self.loss_epoch_eval(module=module, loader_eval=self.setup.loader_val)
        return loss_epoch_train, loss_epoch_val


class Debugger(BatchHandler):
    """
    used for debugging and hyperparameter optimization.
    """
    def __init__(self, setup: Setup):
        super(Debugger, self).__init__(setup=setup)

    def overfit_batch(self,
                      module: Module,
                      batch_debug: Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]],
                      n_iters: int,
                      lrrt_loader: DataLoader = None,
                      lr: float = 1e-5,
                      freeze_pretrained: bool = False) -> Tuple[Module, List[float]]:
        """
        overfits one batch to determine if the module can learn.
        used to determine significant bugs in the module structure
        or the data drawing policy.

        :param module: module to debug
        :type module: Module

        :param batch_debug: single batch to debug on
        :type batch_debug: Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]

        :param n_iters: determines how many iterations the module is trained on the debug batch
        :type n_iters: int

        :param lrrt_loader: loader for determining a learning rate using lrrt. lrrt is not used if None
        :type lrrt_loader: DataLoader

        :param lr: learning rate that is used when lrrt is not used
        :type lr: float

        :param freeze_pretrained: determines if pretrained layers are frozen
        :type freeze_pretrained: bool

        :return: module, batch losses
        :rtype: Tuple[Module, List[float]]
        """
        if lrrt_loader is not None:
            lr, _ = self.lrrt(loader=lrrt_loader, freeze_pretrained=freeze_pretrained)
        optimizer = self.setup.optimizer_class(params=module.parameters(), lr=lr)

        module.train()
        losses = []
        for _ in range(n_iters):
            losses.append(self.train_batch(module=module, optimizer=optimizer, batch=batch_debug,
                                           freeze_pretrained=freeze_pretrained))
        return module, losses


class Visualizer:
    """
    used for standard plots
    """
    @staticmethod
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

    @staticmethod
    def lines_multiplot(lines: List[List[float]],
                        title: str,
                        multiplot_labels: List[str],
                        y_label: str,
                        x_label: str,
                        file_name: str):
        """
        creates multiple lines in the same subplot

        :param lines: float representations of lines to plot
        :type lines: List[List[float]]

        :param title: figure title
        :type title: str

        :param multiplot_labels: line labels
        :type multiplot_labels: List[List[str]]

        :param y_label: y label
        :type y_label: str

        :param x_label: x label
        :type x_label: str

        :param file_name: name of the file in which the figure is stored
        :type file_name: str
        """
        plt.figure(figsize=(4, 4))
        for i, line in enumerate(lines):
            plt.plot(range(len(line)), line, label=multiplot_labels[i])
        plt.title(title)
        plt.ylabel = y_label
        plt.xlabel = x_label
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"../monitoring/{file_name}.png")

    def lines_subplot(self,
                      lines: List[List[float]],
                      title: str,
                      subplot_titles: List[str],
                      y_label: str,
                      x_label: str,
                      file_name: str):
        """
        creates multiple lines in individual subplots

        :param lines: float representations of lines to plot
        :type lines: List[List[float]]

        :param title: figure title
        :type title: str

        :param subplot_titles: line labels
        :type subplot_titles: List[List[str]]

        :param y_label: y label
        :type y_label: str

        :param x_label: x label
        :type x_label: str

        :param file_name: name of the file in which the figure is stored
        :type file_name: str
        """
        n_lines = len(lines)
        n_cols = self.largest_divisor(n=n_lines)
        n_rows = n_lines // n_cols
        plt.figure(figsize=(n_cols*4, n_rows*4))
        plt.suptitle(title)
        for i in range(n_lines):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.title(subplot_titles[i])
            plt.ylabel = y_label
            plt.xlabel = x_label
            plt.plot(range(len(lines[i])), lines[i])
        plt.tight_layout()
        plt.savefig(f"../monitoring/{file_name}.png")


class Trainer(BatchHandler):
    """
    used for training
    """
    def __init__(self, setup: Setup):
        super(Trainer, self).__init__(setup=setup)

    def train(self,
              max_epochs: int,
              freeze_pretrained: bool = False) -> Tuple[Module, List[float], List[float], List[float]]:
        """
        training procedure
        can perform learning rate range test and early stopping if specified in the setup.

        :param max_epochs: maximum amount of training epochs
        :type max_epochs: int

        :param freeze_pretrained: determines if pretrained layers are frozen
        :type freeze_pretrained: bool

        :return: trained module, epoch learning rates, epoch train losses, epoch validation losses
        :rtype: List[Module, List[float], List[float], List[float]]
        """
        es_violations = 0
        losses_train = []
        losses_val = []
        best_lrs = []

        module = load_secure(self.setup.checkpoint_running)
        loss_train, loss_val_last = self.losses_epoch_eval(module=module)
        print("initial eval loss val", loss_val_last, "initial eval loss train", loss_train)
        losses_train.append(loss_train)
        losses_val.append(loss_val_last)
        best_lrs.append(0)

        for epoch in range(1, max_epochs + 1):
            print("training epoch", epoch)

            # lrrt
            lr_best, _ = self.lrrt(freeze_pretrained=freeze_pretrained,
                                   loader=self.setup.loader_train)

            # epoch training
            module = load_secure(self.setup.checkpoint_running).to(self.setup.device)
            optimizer = self.setup.optimizer_class(params=module.parameters(), lr=lr_best)
            self.train_n_batches(module=module,
                                 optimizer=optimizer,
                                 n_batches=len(self.setup.loader_train),
                                 freeze_pretrained=freeze_pretrained,
                                 loader=self.setup.loader_train)

            # epoch evaluation
            loss_train, loss_val = self.losses_epoch_eval(module=module)
            print("eval loss val", loss_val, "eval loss train", loss_train)
            losses_train.append(loss_train)
            losses_val.append(loss_val)
            best_lrs.append(lr_best)

            # early stopping checkpointing
            if loss_val < loss_val_last:
                torch.save(module, self.setup.checkpoint_running)
                torch.save(module, self.setup.checkpoint_final)
                print("loss improvement achieved, final checkpoint updated")
                loss_val_last = loss_val
                es_violations = 0
            else:
                es_violations += 1
                torch.save(module, self.setup.checkpoint_running)
                print("no loss improvement, es violations:", es_violations, "of", self.setup.es_max_violations)
                if es_violations == self.setup.es_max_violations:
                    print("early stopping")
                    break
        return load_secure(self.setup.checkpoint_final), best_lrs, losses_train, losses_val

    def train_overkill_epoch(self, lr_initial: float, freeze_pretrained: bool) -> Tuple[List[float], float, int]:
        """
        one epoch of overkill training
        for each batch, overkill training decays the learning rate until an improvement
        on the epoch validation loss is achieved or a maximum number of decays is reached.
        only updates the running checkpoint if an improvement is achieved.
        this can (and most likely will) overfit the validation data leading to worse performance on test

        :param lr_initial: initial learning rate
        :type lr_initial: float

        :param freeze_pretrained: determines if pretrained layers are frozen
        :type freeze_pretrained: bool

        :return: list of used learning rates, last (updated) validation loss, #batches on which no update was performed
        :rtype: List[List[float], float, int]
        """
        violation_counter = len(self.setup.loader_train)
        best_lrs = []
        module = load_secure(self.setup.checkpoint_running)
        loss_val_last = self.loss_epoch_eval(module=module, loader_eval=self.setup.loader_val)
        for i, batch in enumerate(self.setup.loader_train):
            print("iter", i+1, "of", len(self.setup.loader_train))
            lr = lr_initial
            for _ in range(self.setup.overkill_max_decays + 1):
                module = load_secure(self.setup.checkpoint_running)
                optimizer = self.setup.optimizer_class(params=module.parameters(), lr=lr)
                self.train_batch(module=module, optimizer=optimizer, batch=batch, freeze_pretrained=freeze_pretrained)
                loss = self.loss_epoch_eval(module=module, loader_eval=self.setup.loader_val)
                if loss < loss_val_last:
                    print("improvement, loss", loss)
                    loss_val_last = loss
                    torch.save(module, self.setup.checkpoint_running)
                    best_lrs.append(lr)
                    violation_counter -= 1
                    break
                lr = lr * self.setup.overkill_decay
        return best_lrs, loss_val_last, violation_counter

    def train_overkill(self,
                       max_epochs: int,
                       freeze_pretrained: bool = False) -> Tuple[Module, List[float], List[float], List[float]]:
        """
        performs multiple epochs of overkill training.
        early stops when less than a desired amount of updates was performed per epoch.
        in each epoch, the initial learning rate is adapted to the optimal learning rates in the
        previous epoch.
        see train_overkill_epoch for more details.

        :param max_epochs: max number of performed training epochs
        :type max_epochs: int

        :param freeze_pretrained: determines if pretrained layers are frozen
        :type freeze_pretrained: bool

        :return: trained module, learning rates that lead to improvement, train epoch losses, validation epoch losses
        :rtype: List[Module, List[float], List[float], List[float]]
        """
        losses_train = []
        losses_val = []
        best_lrs = []
        lr = self.setup.overkill_initial_lr
        module = load_secure(self.setup.checkpoint_running)
        losses_train.append(self.loss_epoch_eval(module=module, loader_eval=self.setup.loader_train))
        losses_val.append(self.loss_epoch_eval(module=module, loader_eval=self.setup.loader_val))
        for epoch in range(max_epochs):
            print("epoch", epoch + 1)
            best_lrs_epoch, loss_val_epoch, violations = self.train_overkill_epoch(freeze_pretrained=freeze_pretrained,
                                                                                   lr_initial=lr)
            best_lrs.append(best_lrs_epoch)
            losses_val.append(loss_val_epoch)
            module = load_secure(self.setup.checkpoint_running)
            losses_train.append(self.loss_epoch_eval(module=module, loader_eval=self.setup.loader_train))
            print("train loss:", losses_train[-1])
            print("val loss:", losses_val[-1])
            lr = float(torch.tensor(best_lrs[-1]).mean()) * 1.25
            if violations >= self.setup.overkill_max_violations:
                print("too few updates in this epoch, convergence")
                break
        torch.save(module, self.setup.checkpoint_final)
        return module, list(itertools.chain(*best_lrs)), losses_train, losses_val

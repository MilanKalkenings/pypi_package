import copy

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from typing import Union, List, Tuple

from . modules import Module, ModuleOutput
from . utils import load_secure, save_secure
from . visualization import lines_subplot


class Trainer:
    def __init__(self,
                 module: Module,
                 loader_train: DataLoader,
                 loader_val: DataLoader,
                 loader_test: DataLoader,
                 optimizer_class,
                 device: str = "cpu",
                 checkpoint_running: str = "../monitoring/checkpoint_running.pkl",
                 checkpoint_final: str = "../monitoring/checkpoint_final.pkl"):
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.loader_test = loader_test
        self.optimizer_class = optimizer_class
        self.device = device
        self.checkpoint_running = checkpoint_running
        self.checkpoint_final = checkpoint_final
        save_secure(module=module, file=checkpoint_running)

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
        x = x.to(self.device)
        y = y.to(self.device)
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
        optimizer.zero_grad()
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
                        freeze_pretrained: bool) -> List[float]:
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
        return losses

    def predict_epoch_scores_eval(self, module: Module, loader_eval: DataLoader) -> torch.Tensor:
        """
        performs predictions on all observations in loader_eval and returns the scores
        in eval mode

        :param module: module to be evaluated
        :type module: Module

        :param loader_eval: data loader used for evaluation, should use a SequentialSampler in best case
        :type loader_eval: DataLoader

        :return: scores
        :rtype: torch.Tensor
        """
        scores = []

        for batch in loader_eval:
            scores.append(self.forward_batch(module=module, batch=batch)["scores"].detach_())
        return torch.cat(scores)

    def loss_epoch_eval(self, module: Module, loader_eval: DataLoader) -> float:
        """
        calculates the total epoch loss in evaluation mode

        :param module: module to be evaluated
        :type module: Module

        :param loader_eval: data loader used for evaluation
        :type loader_eval: DataLoader

        :return: epoch loss
        :rtype: float
        """
        batch_losses = torch.zeros(size=[len(loader_eval)])
        for batch_nr, batch in enumerate(loader_eval):
            batch_losses[batch_nr] = self.loss_batch_eval(module=module, batch=batch)
        return float(batch_losses.sum())

    def losses_epoch_eval(self, module: Module) -> Tuple[float, float]:
        """
        wrapper for loss_epoch_eval
        calculates epoch loss for training data and validation data

        :param module: module to be evaluated
        :type module: Module

        :return: training epoch loss, validation epoch loss
        :rtype: List[float, float]
        """
        loss_epoch_train = self.loss_epoch_eval(module=module, loader_eval=self.loader_train)
        loss_epoch_val = self.loss_epoch_eval(module=module, loader_eval=self.loader_val)
        return loss_epoch_train, loss_epoch_val

    def overfit_batch(self,
                      module: Module,
                      batch_debug: Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]],
                      n_iters: int,
                      lr: float = 1e-5,
                      freeze_pretrained: bool = False) -> Tuple[Module, List[float]]:
        """
        overfits one batch to determine if the module can learn.
        used to determine significant bugs

        :param module: module to debug
        :type module: Module

        :param batch_debug: single batch to debug on
        :type batch_debug: Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]]

        :param n_iters: determines how many iterations the module is trained on the debug batch
        :type n_iters: int

        :param lr: learning rate that is used
        :type lr: float

        :param freeze_pretrained: determines if pretrained layers are frozen
        :type freeze_pretrained: bool

        :return: module, batch losses
        :rtype: Tuple[Module, List[float]]
        """
        optimizer = self.optimizer_class(params=module.parameters(), lr=lr)

        module.train()
        losses = []
        for _ in range(n_iters):
            losses.append(self.train_batch(module=module, optimizer=optimizer, batch=batch_debug,
                                           freeze_pretrained=freeze_pretrained))
        return module, losses

    def train(self,
              n_epochs: int,
              lrs: List[float],
              early_stopping_max_violations: int = None,
              freeze_pretrained: bool = False,
              printouts: bool = True) -> Tuple[Module, List[float], List[float]]:
        """
        training procedure

        :param n_epochs: maximum amount of training epochs
        :type n_epochs: int

        :param lrs: learning rates, one per epoch
        :type lrs: List[float]

        :param early_stopping_max_violations: maximum amount of subsequent early stopping violations until the training
        is stopped. No early stopping is performed when early_stopping_max_violations = None

        :type early_stopping_max_violations: int

        :param freeze_pretrained: determines if pretrained layers are frozen
        :type freeze_pretrained: bool

        :param printouts: if True, progress is printed
        :type printouts: bool

        :return: trained module, epoch train losses, epoch validation losses
        :rtype: List[Module, List[float], List[float]]
        """
        early_stopping_violations = 0
        losses_train = []
        losses_val = []

        module = load_secure(self.checkpoint_running)

        # init optimizer here and change arguments only, so that gradient history is kept throughout epochs
        optimizer = self.optimizer_class(params=module.parameters(), lr=lrs[0])
        loss_train, loss_val_last = self.losses_epoch_eval(module=module)
        if printouts:
            print("eval loss val", loss_val_last, "eval loss train", loss_train)
        losses_train.append(loss_train)
        losses_val.append(loss_val_last)

        for epoch in range(1, n_epochs + 1):
            if printouts:
                print("training epoch", epoch)
            # epoch training
            for pg in optimizer.param_groups:
                pg["lr"] = lrs[epoch - 1]

            self.train_n_batches(module=module,
                                 optimizer=optimizer,
                                 n_batches=len(self.loader_train),
                                 freeze_pretrained=freeze_pretrained,
                                 loader=self.loader_train)

            # epoch evaluation
            loss_train, loss_val = self.losses_epoch_eval(module=module)
            if printouts:
                print("eval loss val", loss_val, "eval loss train", loss_train)
            losses_train.append(loss_train)
            losses_val.append(loss_val)

            # early stopping checkpointing
            if loss_val < loss_val_last:
                save_secure(module=module, file=self.checkpoint_running)
                save_secure(module=module, file=self.checkpoint_final)
                if printouts:
                    print("loss improvement achieved, final checkpoint updated")
                loss_val_last = loss_val
                early_stopping_violations = 0
            else:
                early_stopping_violations += 1
                save_secure(module=module, file=self.checkpoint_running)
                if printouts:
                    print("no loss improvement, es violations:", early_stopping_violations, "of", early_stopping_max_violations)
                if early_stopping_violations == early_stopping_max_violations:
                    if printouts:
                        print("early stopping")
                    break
        return load_secure(self.checkpoint_final), losses_train, losses_val

    def acc_epoch_eval(self, module: Module, loader_eval: DataLoader) -> float:
        module.eval()
        preds = []
        labels = []
        for batch in loader_eval:
            scores = self.forward_batch(module=module, batch=batch)["scores"]
            preds.append(torch.argmax(scores, dim=1))
            labels.append(batch[1].to(self.device))
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        return float(torch.sum(preds == labels) / len(labels))

    def determine_initial_lr(self, module: Module, n_iters: int, lr_candidates: List[float], batch_debug: Tuple[Union[torch.Tensor, List[torch.Tensor]], Union[torch.Tensor, List[torch.Tensor]]], save_file: str, freeze_pretrained: bool = False) -> None:
        all_losses = []
        for lr_candidate in lr_candidates:
            _, losses =self.overfit_batch(module=copy.deepcopy(module),
                                          batch_debug=batch_debug,
                                          n_iters=n_iters,
                                          lr=lr_candidate,
                                          freeze_pretrained=freeze_pretrained)
            all_losses.append(losses)
        lines_subplot(lines=all_losses,
                      title="lr debugging",
                      subplot_titles=["lr " + str(lr) for lr in lr_candidates],
                      x_label="iteration",
                      y_label="loss",
                      save_file=save_file)

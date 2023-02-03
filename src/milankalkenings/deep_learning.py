from abc import ABC, abstractmethod
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


class Module(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def freeze_pretrained_layers(self):
        pass

    @abstractmethod
    def unfreeze_pretrained_layers(self):
        pass

    @abstractmethod
    def forward(self, x, y):
        """
        :return: dict {"loss": , "scores", ...}
        """
        pass


def make_reproducible(seed: int = 1):
    """
    ensures reproducibility over multiple script runs and after restarting the local machine
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
    print("reproducible with seed", seed)


class TrainerSetup:
    def __init__(self,
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
                 es_max_violations: int = 2):

        self.device = device
        self.monitor_n_losses = monitor_n_losses  # prints loss slope after this amount of training steps
        self.checkpoint_initial = checkpoint_initial
        self.checkpoint_running = checkpoint_running
        self.checkpoint_final = checkpoint_final
        self.lrrt_n_batches = lrrt_n_batches  # batches used in lrrt for learning rate determination
        self.lrrt_slope_desired = lrrt_slope_desired  # exclusive border
        self.lrrt_max_decays = lrrt_max_decays  # max number of candidate decays performed in lrrt
        self.lrrt_decay = lrrt_decay
        self.lrrt_initial_candidates = lrrt_initial_candidates
        self.es_max_violations = es_max_violations  # max number of early stopping violations


class Trainer:
    def __init__(self, loader_train: DataLoader, loader_val: DataLoader, setup: TrainerSetup):
        self.optimizer_class = torch.optim.Adam
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.device = setup.device
        self.monitor_n_losses = setup.monitor_n_losses

        self.checkpoint_initial = setup.checkpoint_initial
        self.checkpoint_running = setup.checkpoint_running
        self.checkpoint_final = setup.checkpoint_final

        self.lrrt_n_batches = setup.lrrt_n_batches
        self.lrrt_slope_desired = setup.lrrt_slope_desired
        self.lrrt_max_decays = setup.lrrt_max_decays
        self.lrrt_decay = setup.lrrt_decay
        self.lrrt_initial_candidates = setup.lrrt_initial_candidates

        self.es_max_violations = setup.es_max_violations

    def forward_batch(self, module: Module, batch: list):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        return module(x=x, y=y)

    def train_batch(self, module: Module, optimizer: Optimizer, batch: list, freeze_pretrained_layers: bool):
        # freeze/unfreeze here: longer runtime, better encapsulation
        if freeze_pretrained_layers:
            module.freeze_pretrained_layers()
        else:
            module.unfreeze_pretrained_layers()
        module.train()
        module.zero_grad()
        loss = self.forward_batch(module=module, batch=batch)["loss"]
        loss.backward()
        optimizer.step()
        return float(loss)

    def train_n_batches(self, module: Module, optimizer: Optimizer, n_batches: int, freeze_pretrained_layers: bool):
        losses = []
        for train_iter, batch in enumerate(self.loader_train):
            if train_iter == n_batches:
                break

            losses.append(self.train_batch(module=module, optimizer=optimizer, batch=batch,
                                           freeze_pretrained_layers=freeze_pretrained_layers))
            if (len(losses) % self.monitor_n_losses) == 0:
                losses_last = np.array(losses[-self.monitor_n_losses:])
                slope_last, _ = np.polyfit(x=np.arange(len(losses_last)), y=losses_last, deg=1)
                print("iter", train_iter + 1, "mean loss", losses_last.mean(), "loss slope", slope_last)
        slope_total, bias_total = np.polyfit(x=np.arange(len(losses)), y=losses, deg=1)
        return losses, slope_total, bias_total

    def loss_batch_eval(self, module: Module, batch: list):
        module.eval()
        with torch.no_grad():
            return float(self.forward_batch(module=module, batch=batch)["loss"])

    def loss_epoch_eval(self, module: Module, loader_eval: DataLoader):
        batch_losses = np.zeros(len(loader_eval))
        for batch_nr, batch in enumerate(loader_eval):
            batch_losses[batch_nr] = self.loss_batch_eval(module=module, batch=batch)
        return float(batch_losses.mean())

    def losses_epoch_eval(self, module: Module):
        loss_epoch_train = self.loss_epoch_eval(module=module, loader_eval=self.loader_train)
        loss_epoch_val = self.loss_epoch_eval(module=module, loader_eval=self.loader_val)
        return loss_epoch_train, loss_epoch_val

    def predict_class_labels_batch(self, module: Module, batch: list):
        scores = self.forward_batch(module=module, batch=batch)["scores"]
        return torch.argmax(scores, dim=1)

    def overfit_one_train_batch(self, module: Module, batch: list, optimizer: Optimizer, n_iters: int, freeze_pretrained_layers: bool):
        module.train()
        losses = []
        for iter in range(n_iters):
            losses.append(self.train_batch(module=module, optimizer=optimizer, batch=batch,
                                           freeze_pretrained_layers=freeze_pretrained_layers))
        return module, losses

    def lrrt(self, freeze_pretrained_layers: bool):
        """
        Learning Rate Range Test; basic idea:
        for each learning rate in a set of learning rate candidates:
            load a checkpoint
            train from the checkpoint on a small amount of batches
            determine the slope of the batch losses
            return the learning rate that creates the steepest negative slope

        modified to rerun with a decayed set of learning rate candidates
        until a max number of iterations or a certain slope is reached.

        :return: best learning rate, best loss slope
        """
        print("lr search using lrrt")
        slope_desired_found = False
        candidate_lrs = self.lrrt_initial_candidates
        lr_best_total = np.inf
        slope_best_total = np.inf
        for decay_it in range(self.lrrt_max_decays + 1):
            candidate_slopes = np.zeros(shape=len(candidate_lrs))
            for i, lr_candidate in enumerate(candidate_lrs):
                module = torch.load(self.checkpoint_running)
                optimizer = self.optimizer_class(params=module.parameters(), lr=lr_candidate)
                candidate_slopes[i] = self.train_n_batches(module=module, optimizer=optimizer, n_batches=self.lrrt_n_batches,
                                                           freeze_pretrained_layers=freeze_pretrained_layers)[1]
            best_candidate_slope_id = np.argmin(candidate_slopes)
            best_candidate_slope = candidate_slopes[best_candidate_slope_id]
            best_candidate_lr = candidate_lrs[best_candidate_slope_id]
            if best_candidate_slope < slope_best_total:
                slope_best_total = best_candidate_slope
                lr_best_total = best_candidate_lr
            if slope_best_total < self.lrrt_slope_desired:
                slope_desired_found = True
                break
            else:
                print("decaying candidate lrs")
                candidate_lrs = candidate_lrs * self.lrrt_decay
        if not slope_desired_found:
            print("lr with desired loss slope", self.lrrt_slope_desired, "not found. using approx best lr instead")
        print("best loss slope", slope_best_total, "best lr", lr_best_total)
        return lr_best_total, slope_best_total

    def train_n_epochs_early_stop_initial_lrrt(self, max_epochs: int, freeze_pretrained_layers: bool):
        """
        determines the initial learning rate per epoch using lrrt.
        early stops (naively after one early stop violation)

        :param int max_epochs: max #training epochs after determining the initial learning rate with lrrt
        :param bool freeze_pretrained_layers:
        :return: early stopped trained module
        """
        es_violations = 0
        module = torch.load(self.checkpoint_running)
        loss_train, loss_val_last = self.losses_epoch_eval(module=module)
        print("initial eval loss val", loss_val_last, "initial eval loss train", loss_train)
        for epoch in range(1, max_epochs + 1):
            print("training epoch", epoch)
            lr_best, _ = self.lrrt(freeze_pretrained_layers=freeze_pretrained_layers)
            module = torch.load(self.checkpoint_running).to(self.device)
            optimizer = self.optimizer_class(params=module.parameters(), lr=lr_best)
            self.train_n_batches(module=module, optimizer=optimizer, n_batches=len(self.loader_train),
                                 freeze_pretrained_layers=freeze_pretrained_layers)

            loss_train, loss_val = self.losses_epoch_eval(module=module)
            print("eval loss val", loss_val, "eval loss train", loss_train)
            if loss_val < loss_val_last:
                torch.save(module, self.checkpoint_running)
                torch.save(module, self.checkpoint_final)
                print("loss improvement achieved, final checkpoint updated")
                loss_val_last = loss_val
                es_violations = 0
            else:
                es_violations += 1
                torch.save(module, self.checkpoint_running)
                print("no loss improvement achieved, early stopping violations:", es_violations, "of", self.es_max_violations)
                if es_violations == self.es_max_violations:
                    print("early stopping")
                    break
        return torch.load(self.checkpoint_final)

    def train_n_epochs_initial_lrrt(self, n_epochs: int, freeze_pretrained_layers: bool):
        """
        determines the initial learning rate per epoch using lrrt.

        :param int n_epochs: #training epochs after determining the initial learning rate with lrrt
        :param bool freeze_pretrained_layers:
        :return: trained module
        """
        module = torch.load(self.checkpoint_running)
        loss_train, loss_val_last = self.losses_epoch_eval(module=module)
        print("initial eval loss val", loss_val_last, "initial eval loss train", loss_train)
        for epoch in range(1, n_epochs + 1):
            print("training epoch", epoch)
            lr_best, _ = self.lrrt(freeze_pretrained_layers=freeze_pretrained_layers)
            module = torch.load(self.checkpoint_running).to(self.device)
            optimizer = self.optimizer_class(params=module.parameters(), lr=lr_best)
            self.train_n_batches(module=module, optimizer=optimizer, n_batches=len(self.loader_train),
                                 freeze_pretrained_layers=freeze_pretrained_layers)

            loss_train, loss_val = self.losses_epoch_eval(module=module)
            print("eval loss val", loss_val, "eval loss train", loss_train)
            torch.save(module, self.checkpoint_running)
            torch.save(module, self.checkpoint_final)
        return torch.load(self.checkpoint_final)

from pytorch_lightning.accelerators.data_parallel import ParallelPlugin
from pytorch_lightning.accelerators.base_plugin import Plugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import AMPType
from typing import Any, Union
import math

import torch

from pytorch_lightning.core import LightningModule
from pytorch_lightning.accelerators.precision import MixedPrecisionPlugin, PrecisionPlugin

from pytorch_lightning.utilities.apply_func import move_data_to_device


class NewAccelerator(object):
    root_device: Union[str, torch.device]

    def __init__(
        self,
        model_ref: LightningModule,
        root_device: Union[str, torch.device],
        precision_plugin: PrecisionPlugin,
        parallel_plugin: ParallelPlugin,
        gradient_clip_val,
    ):
        self.model_ref = model_ref
        self.precision_plugin = precision_plugin
        self.parallel_plugin = parallel_plugin
        self.gradient_clip_val = gradient_clip_val

        self.optimizers = None
        self.lr_schedulers = None
        self.optimizer_frequencies = None
        self.root_device = root_device

    def setup(self, model):
        self.setup_optimizers(model)
        self.connect_plugin(self.precision_plugin)
        self.connect_plugin(self.parallel_plugin)

    def teardown(self):
        pass

    def batch_to_device(self, batch: Any, device: torch.device):
        model = self.model_ref
        if model is not None:
            return model.transfer_batch_to_device(batch, device)
        return move_data_to_device(batch, device)

    def training_step(self, args):
        batch = self.to_device(args[0])

        args[0] = batch

        with self.precision_plugin.train_step_context():
            with self.parallel_plugin.train_step_context():
                return self.model_ref.training_step(*args)

    def validation_step(self, args):
        batch = self.to_device(args[0])

        args[0] = batch

        with self.precision_plugin.val_step_context():
            with self.parallel_plugin.val_step_context():
                return self.model_ref.validation_step(*args)

    def test_step(self, args):
        batch = self.to_device(args[0])

        args[0] = batch

        with self.precision_plugin.test_step_context():
            with self.parallel_plugin.test_step_context():
                return self.model_ref.test_step(*args)

    def process_dataloader(self, dataloader):
        return dataloader

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        return self.precision_plugin.backward(closure_loss, optimizer, opt_idx, *args, **kwargs)

    def optimizer_step(self, optimizer, current_epoch, batch_idx, opt_idx, lambda_closure):
        model_ref = self.model_ref
        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        native_amp = self.trainer.amp_backend == AMPType.NATIVE

        self.precision_plugin.pre_optimizer_step(optimizer)

        # model hook
        model_ref.optimizer_step(
            epoch=current_epoch,
            batch_idx=batch_idx,
            optimizer=optimizer,
            optimizer_idx=opt_idx,
            optimizer_closure=lambda_closure,
            on_tpu=False,  # TPUAccelerator class sets this as True
            using_native_amp=native_amp,
            using_lbfgs=is_lbfgs,
        )

        self.precision_plugin.post_optimizer_step()

    def optimizer_zero_grad(self, current_epoch, batch_idx, optimizer, opt_idx):
        model_ref = self.model_ref
        model_ref.optimizer_zero_grad(current_epoch, batch_idx, optimizer, opt_idx)

    def clip_gradients(self, optimizer, clip_val=None):
        # TODO: separate TPU case from here
        self._clip_gradients(optimizer, clip_val)

    def _clip_gradients(self, optimizer, clip_val=None):
        # use the trainer's clip val if none passed
        grad_clip_val = self.gradient_clip_val
        if clip_val is not None:
            grad_clip_val = clip_val
        grad_clip_val = float(grad_clip_val)

        # this code is a modification of torch.nn.utils.clip_grad_norm_
        # with TPU support based on https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md
        if grad_clip_val <= 0:
            return

        model = self.model_ref

        # TODO: Change this. Probably to isinstance(self.precision_plugin, MixedPrecisionPlugin) and self.precision_plugin.backend == AMPType.APEX
        if self.trainer.amp_backend == AMPType.APEX:
            parameters = self.precision_plugin.master_params(optimizer)
        else:
            parameters = model.parameters()

        max_norm = grad_clip_val
        norm_type = float(2.0)

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))

        device = parameters[0].device

        if norm_type == math.inf:
            total_norm = max(p.grad.data.abs().max() for p in parameters)
        else:
            out = torch.empty(len(parameters), device=device)
            for i, p in enumerate(parameters):
                torch.norm(p.grad.data.to(device), norm_type, out=out[i])
            total_norm = torch.norm(out, norm_type)

        eps = self.precision_plugin.EPSILON

        clip_coef = torch.tensor(max_norm, device=device) / (total_norm + eps)
        clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
        for p in parameters:
            p.grad.data.mul_(clip_coef.to(p.grad.data.device))

    def on_train_epoch_end(self, outputs):
        pass

    def on_train_end(self):
        pass

    # TODO: Check if we can change logic for early stopping to accelerator/trainer completely or have a separate connector (should be self contained)
    def early_stopping_should_stop(self, pl_module):
        return self.trainer.should_stop

    def setup_optimizers(self, model):
        # TODO: Check if we can change logic for early stopping to trainer completely (should be self contained)
        if self.trainer.testing is True:
            return

        optimizers, lr_schedulers, optimizer_frequencies = self.trainer.init_optimizers(model)
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.optimizer_frequencies = optimizer_frequencies

    def connect_plugin(self, plugin: Plugin):
        model, optimizers, schedulers = plugin.connect(
            self.model_ref, self.optimizers, self.lr_schedulers
        )

        self.model_ref = model
        self.optimizers = optimizers
        self.schedulers = schedulers


    def to_device(self, batch):
        return self.batch_to_device(batch, self.root_device)


class NewCPUAccelerator(NewAccelerator):
    def setup(self, model):
        if isinstance(self.precision_plugin, MixedPrecisionPlugin):
            MisconfigurationException("amp + cpu is not supported.  Please use a GPU option")

        if "cpu" not in str(self.root_device):
            raise MisconfigurationException(f"Device should be CPU, got {self.root_device} instead")

        return super().setup(model)


class NewGPUAccelerator(NewAccelerator):
    def setup(self, model):
        if "cuda" not in str(self.root_device):
            raise MisconfigurationException(f"Device should be GPU, got {self.root_device} instead")
        torch.cuda.set_device(self.root_device)
        self.model_ref.to(self.root_device)

        return super().setup(model)


# TODO: Add NewTPUAccelerator
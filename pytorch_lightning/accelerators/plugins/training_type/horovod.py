from contextlib import ExitStack
from pytorch_lightning.utilities.distributed import rank_zero_only
from typing import Any, List, Optional, Union

import torch
from pytorch_lightning.accelerators.plugins.training_type.parallel import ParallelPlugin
from pytorch_lightning.utilities import _HOROVOD_AVAILABLE
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim.lr_scheduler import _LRScheduler

if _HOROVOD_AVAILABLE:
    import horovod.torch as hvd

if torch.distributed.is_available():
    from torch.distributed import ReduceOp
else:

    class ReduceOp:
        SUM = None


class HorovodPlugin(ParallelPlugin):
    def __init__(self, parallel_devices: List[torch.device]):
        super().__init__(parallel_devices=parallel_devices, cluster_environment=None)

    @property
    def root_device(self):
        return self.parallel_devices[self.local_rank]

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=hvd.size(), rank=hvd.rank())
        return distributed_sampler_kwargs

    def setup(self, model):
        self._model = model

        self.global_rank = hvd.rank()
        self.local_rank = hvd.local_rank()
        rank_zero_only.rank = self.global_rank

        self.model_to_device()

    def pre_training(self):
        def _unpack_lightning_optimizer(opt):
            return opt._optimizer if isinstance(opt, LightningOptimizer) else opt

        optimizers = self.lightning_module.trainer.optimizers
        optimizers = [_unpack_lightning_optimizer(opt) for opt in optimizers]

        # Horovod: scale the learning rate by the number of workers to account for
        # increased total batch size
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= hvd.size()

        # Horovod: adjust base LR used by schedulers to match scaled optimizer initial LR
        lr_schedulers = self.lightning_module.trainer.lr_schedulers
        for scheduler in lr_schedulers:
            scheduler = scheduler["scheduler"]
            if isinstance(scheduler, _LRScheduler):
                scheduler.base_lrs = [lr * hvd.size() for lr in scheduler.base_lrs]

        # Horovod: broadcast parameters & optimizer state to ensure consistent initialization
        hvd.broadcast_parameters(self.lightning_module.state_dict(), root_rank=0)
        for optimizer in optimizers:
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        def _filter_named_parameters(model, optimizer):
            opt_params = set([p for group in optimizer.param_groups for p in group.get("params", [])])
            return [(name, p) for name, p in model.named_parameters() if p in opt_params]

        # Horovod: wrap optimizers to perform gradient aggregation via allreduce
        optimizers = [
            hvd.DistributedOptimizer(
                optimizer, named_parameters=_filter_named_parameters(self.lightning_module, optimizer)
            )
            for optimizer in optimizers
        ]

        optimizers = self.lightning_module.trainer.convert_to_lightning_optimizers(optimizers)
        self.lightning_module.trainer.optimizers = optimizers

    def start_training(self, trainer):
        with ExitStack() as stack:
            for optimizer in trainer.optimizers:
                # Synchronization will be performed explicitly following backward()
                stack.enter_context(optimizer.skip_synchronize())

            # set up training routine
            self._results = trainer.train()

        # Make sure all workers have finished training before returning to the user
        hvd.join()

    def start_testing(self, trainer):
        with ExitStack() as stack:
            # set up training routine
            # self.trainer.train_loop.setup_training(self.trainer.model)
            self._results = trainer.run_test()

        # Make sure all workers have finished training before returning to the user
        hvd.join()

    def barrier(self, *args, **kwargs):
        hvd.join()

    def broadcast(self, obj: object, src: int = 0) -> object:
        obj = hvd.broadcast_object(obj, src)
        return obj

    def model_to_device(self):
        if self.on_gpu:
            torch.cuda.set_device(self.root_device)
        self.model.to(self.root_device)

    def reduce(self, output, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None):
        if group is not None:
            raise ValueError(
                "Horovod does not support allreduce using a subcommunicator at this time. " "Unset `group`."
            )

        if reduce_op is None or reduce_op == "sum":
            reduce_op = hvd.Sum
        elif isinstance(reduce_op, str) and reduce_op in ("avg", "mean"):
            reduce_op = hvd.Average
        else:
            raise ValueError(f"unrecognized `reduce_op`: {reduce_op}")

        # sync all processes before reduction
        hvd.join()
        return hvd.allreduce(output, op=reduce_op)

    def gather_all_tensors(self, result: Union[torch.Tensor], group: Optional[Any] = None):
        if group is not None:
            raise ValueError(
                "Horovod does not support allgather using a subcommunicator at this time. " "Unset `group`."
            )

        if len(result.shape) == 0:
            # Convert scalars to single dimension tensors
            result = result.reshape(1)

        # sync and gather all
        hvd.join()
        gathered = hvd.allgather(result)
        gathered_result = list(gathered.split(1, dim=0))
        return gathered_result

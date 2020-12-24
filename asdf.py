import torch

from pytorch_lightning.accelerators.data_parallel import DDPSpawnPlugin

s = DDPSpawnPlugin(
    parallel_devices=[torch.device("cuda", 0), torch.device("cuda", 1)]
)
s.setup(None)
torch.save(s, "here.ckpt")
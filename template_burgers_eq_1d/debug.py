from pinn import PINN
from pinn import MLP
import torch

pinn = PINN()
pinn.num_epochs = 100
pinn.train()
error = pinn.compute_l2_error()
print(error)
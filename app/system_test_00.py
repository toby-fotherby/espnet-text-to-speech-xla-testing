import torch
import torch_xla
import torch_xla.core.xla_model as xm

print(torch.__version__)
print(torch_xla.__version__)

t = torch.randn(2, 2, device=xm.xla_device())
print(t.device)
print(t)

t0 = torch.randn(2, 2, device=xm.xla_device())
t1 = torch.randn(2, 2, device=xm.xla_device())
print(t0 + t1)

l_in = torch.randn(10, device=xm.xla_device())
linear = torch.nn.Linear(10, 20).to(xm.xla_device())
l_out = linear(l_in)
print(l_out)
import torch
from torch.nn import Parameter
optimizer = torch.optim.SGD([Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
for i in range(10):
    print(i, scheduler.get_last_lr())
    scheduler.step()
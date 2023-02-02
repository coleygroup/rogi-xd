import torch

from torchdrug import core, datasets, tasks, models


# dataset = datasets.ZINC250k("data/")
dataset = datasets.ClinTox("data/", atom_feature="pretrain", bond_feature="pretrain")

hidden_dims = [300, 300, 300, 300, 300]

model = models.GIN(
    dataset.node_feature_dim,
    hidden_dims,
    dataset.edge_feature_dim,
    batch_norm=True,
    readout="mean"
)
task = tasks.AttributeMasking(model, mask_rate=0.15)

optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, dataset, None, None, optimizer, gpus=[0], batch_size=256)

solver.train(num_epoch=10)
solver.save("gin_attr-mask_zinc250k.pt")
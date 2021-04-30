from argparse import ArgumentParser

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from pl_examples.repro import BoringModel, RandomDataset
from pytorch_lightning import seed_everything


def train():
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    device_ids = list(range(args.gpus))
    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(args.local_rank)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    model = BoringModel(**vars(args)).to(device)
    opt = model.configure_optimizers()

    ddp_model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    dataset = RandomDataset(32, 6400)
    train_data = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(dataset, num_replicas=args.gpus, rank=args.local_rank)
    )

    for epoch in range(5):
        for i, batch in enumerate(train_data):
            batch = batch.to(device)
            opt.zero_grad()
            out = ddp_model(batch).sum()
            out.backward()
            opt.step()


if __name__ == "__main__":
    seed_everything(0)
    train()


# run command:
# python -m torch.distributed.launch --nproc_per_node=2 pl_examples/repro_pt.py (--batch_size 4)

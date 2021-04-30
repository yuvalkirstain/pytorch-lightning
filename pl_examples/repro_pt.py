from argparse import ArgumentParser

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from pl_examples.repro import BoringModel, RandomDataset
from pytorch_lightning import seed_everything
import wandb


def train():
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--lr_scale", type=float, default=1.0)
    parser.add_argument("--name", type=str, default="debug")
    args = parser.parse_args()
    device_ids = list(range(args.gpus))
    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(args.local_rank)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        wandb.init(project="ddp-parity-1.3.0", name=args.name)
        wandb.config.update({"gpus": args.gpus, "batch_size" : args.batch_size})

    model = BoringModel(**vars(args)).to(device)
    opt = model.configure_optimizers()

    ddp_model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    dataset = RandomDataset(32, 6400)
    train_data = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(dataset, num_replicas=len(device_ids), rank=args.local_rank)
    )

    global_step = 0
    for epoch in range(5):
        for i, batch in enumerate(train_data):
            batch = batch.to(device)
            opt.zero_grad()
            loss = ddp_model(batch).sum()
            loss.backward()
            opt.step()

            if args.local_rank == 0:
                print(f"{i:04d} / {len(train_data)}")
                wandb.log({"train_loss": loss, "trainer/global_step": global_step})

            global_step += 1


if __name__ == "__main__":
    seed_everything(0)
    train()


# run command:
# python -m torch.distributed.launch --nproc_per_node=2 pl_examples/repro_pt.py --batch_size 4 --gpus 2 --name pt-ddp

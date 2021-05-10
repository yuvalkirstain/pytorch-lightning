import torch.distributed
import time
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--global_rank", type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    print("init")
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://10.10.10.22:1191",
        world_size=2,
        rank=args.global_rank,
    )
    time.sleep(5)

    x = torch.tensor([args.global_rank]).cuda(0)
    print("broadcast", x)
    torch.distributed.broadcast(x, src=0)
    print(x)

    print("barrier")
    torch.distributed.barrier()

    print("after barrier")


if __name__ == "__main__":
    main()

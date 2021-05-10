import os

import torch.distributed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--global_rank", type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    print("init")
    print("rank", int(os.environ.get("RANK", args.global_rank)))
    print("world size", int(os.environ.get("WORLD_SIZE", None)))
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://10.10.10.22:1191",
        world_size=2,
        rank=args.global_rank,
    )
    print("barrier")
    torch.distributed.barrier()


if __name__ == "__main__":
    main()

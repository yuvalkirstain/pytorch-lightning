import os

import torch.distributed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)

    print("init")
    print("rank", int(os.environ["RANK"]))
    print("world size", int(os.environ["WORLD_SIZE"]))
    torch.distributed.init_process_group(
        backend="gloo",
        init_method="env://",
    )
    print("barrier")
    torch.distributed.barrier()


if __name__ == "__main__":
    main()

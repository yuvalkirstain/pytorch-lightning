import os

import torch.distributed


def main():
    print("init")
    print("rank", int(os.environ["RANK"]))
    print("world size", int(os.environ["WORLD_SIZE"]))
    torch.distributed.init_process_group(
        backend="gloo",
        init_method="tcp://10.10.10.22:1191",
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    print("barrier")
    torch.distributed.barrier()


if __name__ == "__main__":
    main()

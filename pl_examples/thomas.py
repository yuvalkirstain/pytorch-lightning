import os

import torch.distributed


def main():
    print("init")
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://10.10.10.22:1191",
        rank=int(os.environ["RANK"]),
        world_size=2
    )
    print("barrier")
    torch.distributed.barrier()


if __name__ == "__main__":
    main()

# barrier_demo.py
import os
import time
import torch.distributed as dist

"""
测试 barrier 方法

OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 -m tests.barrier

- OMP_NUM_THREADS = 每个进程最多使用多少个 CPU 计算线程
"""
def main():
    # dist.init_process_group(backend="nccl") # GPU Backend
    dist.init_process_group(backend="gloo") # CPU Backend

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"rank {rank}: before sleep")
        time.sleep(5)
        print(f"rank {rank}: sleep done")
    else:
        print(f"rank {rank}: no sleep")

    print(f"rank {rank}: before barrier")

    dist.barrier()

    print(f"rank {rank}: after barrier")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

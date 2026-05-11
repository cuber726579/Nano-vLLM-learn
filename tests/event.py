# event_demo.py
import time
import torch.multiprocessing as mp

"""
测试 multiprocessing.Event 的 wait/set/clear。

运行:
python -m tests.event

说明:
- 子进程先启动, 在 event.wait() 处阻塞等待。
- 主进程 sleep 后调用 event.set(), 子进程被唤醒并继续执行。
- 子进程调用 event.clear() 后, 下一次 wait() 会重新阻塞。
"""


def worker(rank: int, event: mp.Event):
    print(f"worker {rank}: start")

    print(f"worker {rank}: wait round 1")
    event.wait()
    print(f"worker {rank}: wake round 1")

    # Event 是一个开关: set 后会一直保持唤醒状态, clear 后才会重新阻塞。
    event.clear()
    print(f"worker {rank}: clear event")

    print(f"worker {rank}: wait round 2")
    event.wait()
    print(f"worker {rank}: wake round 2")


def main():
    ctx = mp.get_context("spawn")
    event = ctx.Event()

    processes = []
    for rank in range(3):
        process = ctx.Process(target=worker, args=(rank, event))
        process.start()
        processes.append(process)

    time.sleep(2)
    print("main: set event round 1")
    event.set()

    time.sleep(2)
    print("main: set event round 2")
    event.set()

    for process in processes:
        process.join()

    print("main: done")


if __name__ == "__main__":
    main()

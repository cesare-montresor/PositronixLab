
from torch import multiprocessing as mp
from time import sleep

def worker(rank, queue):
    print(f"worker {rank} running")
    sleep(rank)

if __name__ == "__main__":
    mp.set_start_method("spawn", True)
    queue = mp.Queue()
    queue.put("init")  # force SemLock materialization before spawn
    ps = []
    for i in range(5):
        p = mp.Process(target=worker, args=(i, queue))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
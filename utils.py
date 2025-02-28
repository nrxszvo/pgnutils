import time
import datetime
from multiprocessing import Lock
import os
import numpy as np
import subprocess


class PrintSafe:
    def __init__(self):
        self.lock = Lock()

    def __call__(self, string, end="\n"):
        self.lock.acquire()
        try:
            print(string, end=end)
        finally:
            self.lock.release()


def timeit(fn):
    start = time.time()
    ret = fn()
    end = time.time()
    nsec = end - start
    hr = int(nsec // 3600)
    minute = int((nsec % 3600) // 60)
    sec = int(nsec % 60)
    return ret, f"{hr}:{minute:02}:{sec:02}"


def get_eta(max_items, items_so_far, start):
    if items_so_far == 0:
        return "tbd"
    end = time.time()
    eta = datetime.timedelta(
        seconds=max((max_items - items_so_far), 0) * (end - start) / items_so_far
    )
    hours = eta.seconds // 3600
    minutes = (eta.seconds % 3600) // 60
    seconds = eta.seconds % 60
    eta_str = f"{eta.days}:{hours:02}:{minutes:02}:{seconds:02}"
    return eta_str


def resize_mmaps(binary, npydir):
    blockDirs = [dn for dn in os.listdir(npydir) if "block-" in dn]
    for dn in blockDirs:
        print(f"resizing {dn}")
        md = np.load(os.path.join(npydir, dn, "md.npy"), allow_pickle=True).item()
        cmd = [
            binary,
            "--blockDir",
            os.path.abspath(os.path.join(npydir, dn)),
            "--ngames",
            str(md["ngames"]),
            "--nmoves",
            str(md["nmoves"]),
        ]
        subprocess.call(cmd)

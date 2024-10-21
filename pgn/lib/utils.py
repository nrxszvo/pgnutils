import time
import datetime


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

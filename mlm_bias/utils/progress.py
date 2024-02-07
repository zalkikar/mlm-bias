#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time

def show_progress(index, total, label, start, len_bar=30, len_out=40):
    prop_complete = (index / total)
    elapsed = time.time() - start
    if index > 0:
        seconds_remaining = ((elapsed / index) * (total - index))
    else:
        seconds_remaining = elapsed * (total - index)
    mins_e, secs_e = divmod(elapsed, 60)
    mins_r, secs_r = divmod(seconds_remaining, 60)
    per1 = int(len_bar * prop_complete)
    per2 = int(len_bar - per1)
    sys.stdout.write('\r')
    sys.stdout.write(
        f"{label} |{per1*'█'}{per2*' '}| "+
        f"{index}/{total} [{int(100.0 * prop_complete)}%] in "+
        f"{f'{int(mins_e)}m ' if mins_e > 0 else ''}{int(secs_e)}s "+
        f"ETA:{f' {int(mins_r)}m' if mins_r > 0 else ''} {int(secs_r)}s".ljust(len_out)
    )
    sys.stdout.flush()

def end_progress():
    sys.stdout.write("\n")
# benchmark.py
# Benchmark prime counting (≤ N) with sequential and multiprocessing-parallel versions.

import time
import csv
import platform
import multiprocessing as mp
from statistics import median

from primenum import count_primes_seq, count_primes_parallel

N = 1_000_000
PROCS = (1, 2, 3, 4, 6, 8, 12)
REPEATS = 3
CSV_PATH = "/Users/d.tummidi/downloads/results.csv"

def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    t1 = time.perf_counter()
    return out, (t1 - t0)

def run_bench():
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    except AttributeError:
        pass

    count_primes_seq(50_000)

    cnt1, T1 = timed(count_primes_seq, N)

    rows = []
    rows.append({
        "mode": "sequential",
        "N": N,
        "p": 1,
        "time": T1,
        "speedup": 1.0,
        "efficiency": 1.0,
        "count": cnt1
    })

    for p in PROCS:
        times = []
        counts = []
        for _ in range(REPEATS):
            c, tp = timed(count_primes_parallel, N, p)
            times.append(tp)
            counts.append(c)

        Tp = median(times)
        cntp = max(set(counts), key=counts.count)  # most frequent count (sanity)
        Sp = T1 / Tp
        Ep = Sp / p

        # Emit a parallel row
        rows.append({
            "mode": "parallel",
            "N": N,
            "p": p,
            "time": Tp,
            "speedup": Sp,
            "efficiency": Ep,
            "count": cntp
        })

        print(f"N={N} p={p} time={Tp:.4f}s speedup={Sp:.2f} efficiency={Ep:.2f} count={cntp}")

    # Write CSV
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "N", "p", "time_s", "speedup", "efficiency", "prime_count", "cpu", "logical_cores"])
        for r in rows:
            w.writerow([
                r["mode"],
                r["N"],
                r["p"],
                f"{r['time']:.6f}",
                f"{r['speedup']:.6f}",
                f"{r['efficiency']:.6f}",
                r["count"],
                platform.processor(),
                mp.cpu_count()
            ])

    print(f"Done writing to the {CSV_PATH}")


if __name__ == "__main__":
    run_bench()
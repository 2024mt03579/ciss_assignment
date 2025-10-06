# primenum.py
import math
from typing import List, Tuple
import multiprocessing as mp

def count_primes_seq(n: int) -> int:
    if n < 2:
        return 0
    flags = bytearray(b"\x01") * (n + 1)
    flags[0] = flags[1] = 0
    root = int(math.isqrt(n))
    for p in range(2, root + 1):
        if flags[p]:
            start = p * p
            step = p
            flags[start:n+1:step] = b"\x00" * ((n - start)//step + 1)
    return sum(flags)

def _sieve_small(limit: int) -> List[int]:
    if limit < 2:
        return []
    flags = bytearray(b"\x01") * (limit + 1)
    flags[0] = flags[1] = 0
    root = int(math.isqrt(limit))
    for p in range(2, root + 1):
        if flags[p]:
            start = p * p
            step = p
            flags[start:limit+1:step] = b"\x00" * ((limit - start)//step + 1)
    return [i for i, v in enumerate(flags) if v]

def _count_segment(args: Tuple[int, int, List[int]]) -> int:
    L, R, small = args
    size = R - L + 1
    flags = bytearray(b"\x01") * size
    # 0/1 not prime
    if L == 0:
        if size >= 1: flags[0] = 0
        if size >= 2: flags[1] = 0
    elif L == 1:
        flags[0] = 0
    for q in small:
        q2 = q * q
        if q2 > R:
            break
        m = q2 if q2 >= L else ((L + q - 1) // q) * q
        for k in range(m, R + 1, q):
            flags[k - L] = 0
    return sum(flags)

def count_primes_parallel(n: int, workers: int) -> int:
    if n < 2:
        return 0
    root = int(math.isqrt(n))
    small = _sieve_small(root)

    total = (n - 1) if n >= 2 else 0  # range [2..n] inclusive
    if total <= 0:
        return 0

    base, extra = divmod(total, workers)
    segments = []
    cur = 2
    for i in range(workers):
        length = base + (1 if i < extra else 0)
        a, b = cur, cur + length - 1
        segments.append((a, b, small))
        cur = b + 1

    with mp.Pool(processes=workers) as pool:
        return sum(pool.map(_count_segment, segments))

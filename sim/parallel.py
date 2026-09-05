"""Running independent experiments across cores.

Every harness here is a loop over experiments that do not depend on each other:
sixteen seeds, twenty-one perturbations, a dozen discount rates. They were run
one at a time on one core, which left eleven idle and made a sensitivity sweep
a forty-five minute wait.

Processes rather than threads, and not only for the GIL. The simulation reseeds
the global `random` module on every run, so two experiments sharing an
interpreter would reset the stream underneath each other -- the same hazard
recovery/jobs.py shells out to avoid. Separate processes each get their own
`random`, which makes parallelism and correctness the same decision here.

That also means a worker cannot mutate a constant for its parent: a cell that
perturbs `core.friction.B_PRICE` sets it inside its own interpreter, and nothing
leaks back. The sequential version had to remember to restore every value in a
`finally` block; this one cannot get that wrong.

Workers are capped below the core count. The box is usually also running the
live simulation behind the dashboard, and saturating every core makes the thing
people are watching stutter.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Leave headroom for the gateway's live simulation and the OS.
RESERVED_CORES = 4
MAX_WORKERS = 8


def worker_count(requested=None) -> int:
    if requested:
        return max(1, int(requested))
    available = (os.cpu_count() or 2) - RESERVED_CORES
    return max(1, min(MAX_WORKERS, available))


def run_cells(worker, cells, workers=None, on_result=None):
    """Map ``worker`` over ``cells`` in parallel, preserving input order.

    ``worker`` must be a module-level function and ``cells`` picklable, because
    each call crosses a process boundary. Results come back in the order the
    cells were given regardless of which finished first, so output tables stay
    readable and reproducible.

    ``on_result`` is called as each cell lands, for progress reporting -- the
    alternative is a silent wait, which is what made the sequential harnesses
    feel broken while they were merely slow.
    """
    cells = list(cells)
    if not cells:
        return []

    count = worker_count(workers)
    if count == 1:
        results = []
        for index, cell in enumerate(cells):
            result = worker(cell)
            results.append(result)
            if on_result:
                on_result(index, cell, result)
        return results

    results = [None] * len(cells)
    with ProcessPoolExecutor(max_workers=count) as pool:
        futures = {pool.submit(worker, cell): index
                   for index, cell in enumerate(cells)}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            if on_result:
                on_result(index, cells[index], results[index])
    return results

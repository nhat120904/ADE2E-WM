"""Run NAVSIM v2.2's official PDM-score entry points with one Ray task per log.

The devkit's ``worker_map`` splits the per-log data points into ``threads_per_node`` contiguous chunks, so
wall time is set by the largest chunk (on navtest with 16 workers one chunk holds 1,221 of 12,146 scenes,
which does not fit in 6 h of CPU inference). Submitting one task per log, largest first, lets Ray hand logs
to idle workers. Each task requests 1 CPU: nuplan's ``Task`` defaults to ``num_cpus=None``, under which Ray
did not cap concurrency and started one ~1.5 GB process per log at once (76 on navhard; on navtest this
ran out of memory). Agent, simulator, scorer, aggregation and CSV writing are the unmodified devkit code;
each scene's score is computed independently, so it does not depend on how logs are scheduled.

usage: python run_pdm_balanced.py {two_stage|one_stage} <hydra overrides...>
"""

import sys

from nuplan.planning.utils.multithreading.worker_pool import Task


def worker_map_per_log(worker, fn, input_objects):
    if worker.number_of_threads == 0:
        return fn(input_objects)
    items = sorted(input_objects, key=lambda a: len(a["tokens"]), reverse=True)
    scattered_objects = worker.map(Task(fn=fn, num_cpus=1), [[a] for a in items])
    return [result for results in scattered_objects for result in results]


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    if mode == "two_stage":
        import navsim.planning.script.run_pdm_score as entry
    elif mode == "one_stage":
        import navsim.planning.script.run_pdm_score_one_stage as entry
    else:
        raise SystemExit(f"unknown mode {mode!r}; expected two_stage or one_stage")
    entry.worker_map = worker_map_per_log
    entry.main()

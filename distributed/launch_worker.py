"""
distributed/launch_worker.py — Start one or more worker processes.

Usage (on any device with market_sim installed)
------------------------------------------------
    # Single worker, training ctrl_id=1
    python -m distributed.launch_worker --server https://xxx.ngrok-free.app

    # Two workers on the same machine (ruler + local)
    python -m distributed.launch_worker --server https://xxx.ngrok-free.app --workers 2

    # Explicit ctrl_ids (ruler only)
    python -m distributed.launch_worker --server https://xxx.ngrok-free.app --ctrl-ids 1

Options
-------
    --server     Parameter server URL (required)
    --workers    Number of parallel worker threads to spawn (default: 1)
    --ctrl-ids   Comma-separated controller IDs to train (default: 1,2)
                 Cycling: if --workers > len(--ctrl-ids), IDs are repeated round-robin.
"""
from __future__ import annotations

import argparse
import logging
import sys
import os
import threading

log = logging.getLogger("dist.launch_worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


def _run_worker(server_url: str, ctrl_id: int):
    """Thread target: create and run one Worker indefinitely."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from distributed.worker import Worker
    try:
        w = Worker(server_url=server_url, ctrl_id=ctrl_id)
        w.run()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        log.error(f"Worker ctrl_id={ctrl_id} crashed: {exc}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Market-Sim distributed worker launcher")
    parser.add_argument("--server",   type=str, required=True,
                        help="Parameter server URL, e.g. https://xxx.ngrok-free.app")
    parser.add_argument("--workers",  type=int, default=1,
                        help="Number of parallel worker threads (default: 1)")
    parser.add_argument("--ctrl-ids", type=str, default="1,2",
                        help="Comma-separated ctrl_ids to train (default: '1,2')")
    args = parser.parse_args()

    ctrl_ids = [int(x.strip()) for x in args.ctrl_ids.split(",")]
    n        = args.workers

    print()
    print("=" * 60)
    print(f"  Starting {n} worker(s)")
    print(f"  Server  : {args.server}")
    print(f"  ctrl_ids: {ctrl_ids}")
    print("=" * 60)
    print()

    threads = []
    for i in range(n):
        ctrl_id = ctrl_ids[i % len(ctrl_ids)]
        t = threading.Thread(
            target=_run_worker,
            args=(args.server, ctrl_id),
            name=f"Worker-{i+1}-ctrl{ctrl_id}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        log.info(f"Started worker thread {i+1}/{n} (ctrl_id={ctrl_id})")

    # Block until all threads finish (they run forever unless interrupted)
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down workers.")


if __name__ == "__main__":
    main()

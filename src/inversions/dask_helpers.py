"""Helper functions for using Dask.

Using of Dask with SLURMCluster to launch SLURM jobs with one Dask worker ("process")
per job.
This means that functions submitted to the client will run on their own jobs if they are not
able to use Dask to parallelise, for instance, running an inversion using PyMC.

>>> from dask_jobqueue import SLURMCluster
>>> from dask.distributed import Client
>>>
>>> cluster = SLURMCluster(
>>>     processes=1,
>>>     cores=8,
>>>     memory='20GB',
>>>     walltime='02:00:00',
>>>     account="abcd123456",
>>>     log_directory=str(log_path),
>>> )
>>> client = Client(cluster)
>>> cluster.scale(jobs=5)

If we have a function ``run_inversion`` and a way of getting arguments for it (in this case,
they're stored in a dictionary), then we can submit multiple runs as follows:

>>> from functools import partial
>>> futures = []
>>>
>>> for year in range(2015, 2020):
>>>     inv_input = inversion_inputs[str(year)]  # inputs from dictionary
>>>     func = partial(run_inversion, inv_input, year)
>>>     future = client.submit(func)
>>>     futures.append(future)

The futures are a reference to the running jobs. The futures are returned immediately,
and we are free to do other work (e.g. in a notebook)

If the function ``run_inversion`` returns ``None``, then Dask will free up resources
once it finishes running.
Otherwise, the future holds the return value of the function, and Dask won't free up
resources until there are no references to the return value (I think?).

"""

import concurrent.futures
from functools import partial
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile

from dask_jobqueue.slurm import SLURMCluster
from dask.distributed import Client, Cluster, Future


def get_available_workers(client: Client, cluster: Cluster) -> list[str]:
    worker_vals = client.scheduler_info(n_workers=len(cluster.workers))[
        "workers"
    ].values()
    available_workers = [v.get("id") for v in worker_vals]
    return available_workers


def get_futures(client: Client) -> list[Future]:
    # TODO: maybe there is a better/"more correct" way to get the keys?
    return [Future(key, client) for key in client.refcount]


def zip_dir_no_compress(
    store_dir: str | Path, zip_path: str | Path, remove_source: bool = False
) -> None:
    store_dir = Path(store_dir)
    zip_path = Path(zip_path)

    if not store_dir.is_dir():
        raise FileNotFoundError(f"Store directory not found: {store_dir}")

    # create temp file in same directory as zip_path (use str(...) for tempfile)
    fd, tmp_name = tempfile.mkstemp(suffix=".zip", dir=str(zip_path.parent))
    os.close(fd)

    try:
        with zipfile.ZipFile(
            tmp_name, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as zf:
            for p in sorted(store_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(store_dir))

        # quick integrity check
        with zipfile.ZipFile(tmp_name, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupt file in zip: {bad}")

        # atomic replace (Path is fine here on modern Python)
        os.replace(tmp_name, str(zip_path))

        if remove_source:
            shutil.rmtree(store_dir)
    except Exception:
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass
        raise


def zip_on_done(
    future: Future, executor: concurrent.futures.ThreadPoolExecutor
) -> None:
    """Zip directory returned by future.

    Example usage:

    >>> executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    >>> for f in futures:
    >>>    f.add_done_callback(zip_on_done, executor)

    """
    try:
        store_dir = future.result()
    except Exception as exc:
        print("Task failed:", exc)
    else:
        store_dir = Path(store_dir)
        zip_path = store_dir.with_suffix(store_dir.suffix + ".zip")

        # Submit heavy I/O (zipping) to a background thread pool
        zip_dir = partial(zip_dir_no_compress, remove_source=True)
        executor.submit(zip_dir, store_dir, zip_path)

import argparse
import importlib
import importlib.metadata
import multiprocessing
import os
import subprocess
import time

import numpy as np
import pynbody
import yaml


def _git_commit() -> str:
    """Return the short git commit hash of the running source checkout, if available."""
    pkg_dir = os.path.dirname(__file__)
    try:
        # A non-editable install just copies this file into site-packages,
        # which may happen to sit inside some unrelated project's git
        # checkout. Only trust the result if this file is actually tracked
        # by the repo found there.
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", os.path.basename(__file__)],
            cwd=pkg_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=pkg_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def print_header() -> None:
    """Print a header with the PEST version and git commit."""
    version = importlib.metadata.version("astro-pest")
    commit = _git_commit()
    title = f"PEST pipeline v{version} ({commit})"
    width = len(title) + 4
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _available_cpus() -> int:
    """Number of CPUs this process may run on, e.g. as allocated by Slurm."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _instantiate(class_path: str, init_args: dict):
    """Instantiate a class from a dotted ``module.ClassName`` string."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**init_args)


class _TransformStep:
    """A single transform (or filter) step.

    With a `column`, the transform receives and returns the value of that
    column. Without a column (`None`), it operates on the whole record: a
    filter receives the record and returns whether to keep it, any other
    transform receives the record and returns the (updated) record.
    """

    def __init__(self, column: str | None, transform):
        self.column = column
        self.transform = transform
        self.is_filter = getattr(transform, "is_filter", False)

    @property
    def name(self) -> str:
        return self.transform.__class__.__name__

    @property
    def label(self) -> str:
        return self.name if self.column is None else f"{self.column}: {self.name}"


def _build_steps(transform_cfgs: list[dict]) -> list[_TransformStep]:
    """Instantiate the transform chain.

    Each config entry is either a column-scoped group
    `{column: ..., transformations: [...]}` or a single record-level step
    `{class_path: ..., init_args: ...}`.
    """
    steps = []
    for cfg in transform_cfgs:
        if "class_path" in cfg:
            steps.append(_TransformStep(None, _instantiate(cfg["class_path"], cfg.get("init_args", {}))))
            continue
        for transform_cfg in cfg.get("transformations", []):
            transform = _instantiate(transform_cfg["class_path"], transform_cfg.get("init_args", {}))
            steps.append(_TransformStep(cfg["column"], transform))
    return steps


def _prefilter(
    dataset, steps: list[_TransformStep], dropped_counts: dict[str, int], step_times: np.ndarray
) -> tuple[np.ndarray, int]:
    """Apply the leading vectorizable filters on the dataset table before extraction.

    Returns the indices of the surviving records and the number of leading
    steps that were applied here and must be skipped per record. The time
    spent in each applied step is added to `step_times`.
    """
    indices = np.arange(len(dataset))
    if not hasattr(dataset, "table"):
        return indices, 0

    table = None
    num_applied = 0
    for i, step in enumerate(steps):
        if not (step.is_filter and step.column is None and hasattr(step.transform, "mask")):
            break
        start = time.perf_counter()
        if table is None:
            table = dataset.table()
        keep = np.asarray(step.transform.mask(table), dtype=bool)[indices]
        if not keep.all():
            dropped_counts[step.name] = dropped_counts.get(step.name, 0) + int((~keep).sum())
        indices = indices[keep]
        step_times[i] += time.perf_counter() - start
        num_applied += 1
    return indices, num_applied


# Populated once per worker process by `_init_worker`, so the dataset and the
# transform chain are each built only once and then reused for every record
# that worker handles.
_worker_state: dict = {}


def _init_worker(
    extract_cfg: dict, transform_cfgs: list[dict], start_step: int = 0, threads_per_worker: int | None = None
) -> None:
    """Pool initializer: build one dataset instance and one transform chain per worker.

    The first `start_step` steps were already applied by `_prefilter` and are skipped.
    """
    if threads_per_worker is not None:
        # pynbody defaults to one thread per CPU of the node in every worker.
        pynbody.config["number_of_threads"] = threads_per_worker
    dataset = _instantiate(extract_cfg["class_path"], extract_cfg.get("init_args", {}))
    steps = _build_steps(transform_cfgs)[start_step:]
    for step in steps:
        if hasattr(step.transform, "bind"):
            step.transform.bind(dataset)
    _worker_state["dataset"] = dataset
    _worker_state["steps"] = steps


def _process_record(index: int) -> tuple[dict | None, str | None, list[float]]:
    """Extract one record and run the full transform chain on it in a single pass.

    Returns ``(record, None, timings)`` when the record survives, or
    ``(None, filter_name, timings)`` naming the filter step that dropped it.
    ``timings`` holds the extraction time followed by the time of each
    executed step.
    """
    dataset = _worker_state["dataset"]
    steps = _worker_state["steps"]
    start = time.perf_counter()
    record = dataset[index]
    timings = [time.perf_counter() - start]

    for step in steps:
        start = time.perf_counter()
        try:
            if step.is_filter:
                if not step.transform(record):
                    timings.append(time.perf_counter() - start)
                    return None, step.name, timings
            elif step.column is None:
                record = step.transform(record)
            else:
                record[step.column] = step.transform(np.array(record[step.column]))
        except Exception as e:
            print(
                f"Transform {step.name} failed for "
                f"simulation={record.get('simulation')}, "
                f"snapshot={record.get('snapshot')}, "
                f"subhalo_id={record.get('subhalo_id')}: {e}"
            )
            raise
        timings.append(time.perf_counter() - start)
    return record, None, timings


class Pipeline:
    def __init__(
        self,
        config: dict,
    ):
        self.config = config
        self.num_workers = config.get("num_workers", 1)
        self.batch_size = config.get("batch_size", 16)
        self.shuffle = config.get("shuffle", True)
        self.seed = config.get("seed", 42)
        # By default, the workers share the available CPUs.
        threads_per_worker = config.get("threads_per_worker")
        if threads_per_worker is None:
            threads_per_worker = max(1, _available_cpus() // max(1, self.num_workers))
        if threads_per_worker < 1:
            raise ValueError(f"threads_per_worker must be at least 1, got {threads_per_worker}")
        self.threads_per_worker = threads_per_worker

    def run(self) -> None:
        """Run the pipeline: extract, transform, and load data.

        Extraction and transformation are fused into a single pass per record:
        each worker loads one record and immediately runs the whole transform
        chain on it, instead of rewriting the full dataset once per step.
        """
        extract_cfg = self.config["extract"]
        transform_cfgs = self.config.get("transform", [])

        # Instantiated once here to get the record count and to apply the
        # leading vectorizable filters on the dataset table; each worker below
        # builds its own instance to avoid pickling/sharing file handles.
        setup_start = time.perf_counter()
        dataset = _instantiate(extract_cfg["class_path"], extract_cfg.get("init_args", {}))
        num_records = len(dataset)
        print(
            f"Extract: {extract_cfg['class_path']} with {num_records} records "
            f"(setup {time.perf_counter() - setup_start:.2f}s)"
        )

        steps = _build_steps(transform_cfgs)
        step_times = np.zeros(len(steps))
        step_calls = np.zeros(len(steps), dtype=int)
        dropped_counts: dict[str, int] = {}
        indices, start_step = _prefilter(dataset, steps, dropped_counts, step_times)
        step_calls[:start_step] = 1
        del dataset

        if start_step:
            print(f"Prefilter: {len(indices)}/{num_records} records remaining")
            for i in range(start_step):
                print(f"  {steps[i].label}: {step_times[i]:.2f}s")

        if self.shuffle:
            np.random.default_rng(self.seed).shuffle(indices)

        print(
            f"Extract+Transform: processing {len(indices)} records with {self.num_workers} worker(s), "
            f"{self.threads_per_worker} thread(s) each"
        )
        transform_start = time.perf_counter()
        if self.num_workers > 1:
            with multiprocessing.Pool(
                self.num_workers,
                initializer=_init_worker,
                initargs=(extract_cfg, transform_cfgs, start_step, self.threads_per_worker),
            ) as pool:
                results = list(pool.imap(_process_record, indices, chunksize=max(1, self.batch_size)))
        else:
            _init_worker(extract_cfg, transform_cfgs, start_step, self.threads_per_worker)
            results = [_process_record(i) for i in indices]

        records = []
        extract_time = 0.0
        for record, dropped_by, timings in results:
            if record is not None:
                records.append(record)
            else:
                dropped_counts[dropped_by] = dropped_counts.get(dropped_by, 0) + 1
            extract_time += timings[0]
            num_executed = len(timings) - 1
            step_times[start_step : start_step + num_executed] += timings[1:]
            step_calls[start_step : start_step + num_executed] += 1
        del results

        print(
            f"Extract+Transform: kept {len(records)}/{num_records} records "
            f"in {time.perf_counter() - transform_start:.2f}s"
        )
        for name, count in dropped_counts.items():
            print(f"  dropped by {name}: {count}")

        # Per-step times are summed over all records (and thus over all
        # workers), so with multiple workers they exceed the wall time above.
        print("Time per step (summed over workers):")
        label_width = max([len("extract")] + [len(step.label) for step in steps])
        self._print_step_time("extract", extract_time, len(indices), label_width)
        for i in range(start_step, len(steps)):
            self._print_step_time(steps[i].label, step_times[i], step_calls[i], label_width)

        # Load
        load_start = time.perf_counter()
        load_cfgs = self.config.get("load", [])
        loads = [_instantiate(cfg["class_path"], cfg.get("init_args", {})) for cfg in load_cfgs]

        load_times = []
        for load in loads:
            start = time.perf_counter()
            load(records)
            load_times.append((load.__class__.__name__, time.perf_counter() - start))
        print(f"Load: {len(records)} records in {time.perf_counter() - load_start:.2f}s")
        for name, seconds in load_times:
            print(f"  {name}: {seconds:.2f}s")

    @staticmethod
    def _print_step_time(label: str, total: float, calls: int, label_width: int) -> None:
        per_record = f"{1e3 * total / calls:.3f} ms/record" if calls else "-"
        print(f"  {label:<{label_width}}  {total:9.2f}s  {calls:8d} calls  {per_record}")


def main() -> None:
    """CLI entry point: read a YAML config file and run the pipeline."""

    print_header()

    parser = argparse.ArgumentParser(
        prog="pest",
        description="Preprocessing Engine for Spherinator Training",
    )
    parser.add_argument("config", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    print(f"Config: {os.path.abspath(args.config)}")
    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    Pipeline(config).run()


if __name__ == "__main__":
    main()

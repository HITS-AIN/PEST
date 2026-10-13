"""Benchmark RenderStars against the number of star particles.

Synthetic exponential-disk galaxies are built in memory with `pynbody.new`, so
no snapshot on disk is needed. Every render gets a freshly built snapshot (as
`PynbodyDataset.halo` returns a standalone copy), so cached arrays such as the
smoothing lengths are not reused. Building the snapshot is not timed.

Run `uv run python benchmarks/render_stars.py --help` for the options.
"""

import argparse
import time

import numpy as np
import pynbody

from pest import RenderStars

STELLAR_MASS = 5e10  # [Msun]


def make_galaxy(n_stars: int, seed: int = 0):
    """A rotating exponential disk with random stellar ages and metallicities."""
    rng = np.random.default_rng(seed)
    sim = pynbody.new(star=n_stars)
    r = rng.exponential(3.0, n_stars)  # [kpc]
    phi = rng.uniform(0, 2 * np.pi, n_stars)
    z = rng.normal(0, 0.3, n_stars)
    # Offset from the origin, so that the centering has to do some work.
    sim.s["pos"] = np.c_[r * np.cos(phi), r * np.sin(phi), z] + 5000.0
    sim.s["pos"].units = "kpc"
    v_circ = 200.0 * (1 - np.exp(-r / 2.0))  # [km/s]
    sim.s["vel"] = np.c_[-v_circ * np.sin(phi), v_circ * np.cos(phi), rng.normal(0, 20, n_stars)]
    sim.s["vel"].units = "km s^-1"
    sim.s["mass"] = np.full(n_stars, STELLAR_MASS / n_stars)
    sim.s["mass"].units = "Msol"
    sim.s["tform"] = rng.uniform(0.1, 13.5, n_stars)
    sim.s["tform"].units = "Gyr"
    sim.s["metals"] = rng.uniform(0.002, 0.03, n_stars)
    sim.properties["time"] = pynbody.units.Unit("13.8 Gyr")
    sim.properties["a"] = 1.0
    sim.properties["h"] = 0.6774
    return sim


class PrebuiltDataset:
    """Hands out a snapshot built in advance, so that only the rendering is timed."""

    def __init__(self):
        self.snapshot = None

    def halo(self, subhalo_id):
        return self.snapshot


def benchmark(n_stars: int, repeat: int, step: RenderStars, dataset: PrebuiltDataset) -> np.ndarray:
    times = []
    for i in range(repeat + 1):
        dataset.snapshot = make_galaxy(n_stars, seed=i)
        start = time.perf_counter()
        step({"subhalo_id": i})
        times.append(time.perf_counter() - start)
    return np.array(times[1:])  # drop the warm-up run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--n-stars",
        type=int,
        nargs="+",
        default=[1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000],
    )
    parser.add_argument("--repeat", type=int, default=3, help="timed runs per size, after one warm-up run")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--width", type=float, default=50.0, help="image width [kpc]")
    parser.add_argument("--center-mode", default="ssc")
    args = parser.parse_args()

    dataset = PrebuiltDataset()
    step = RenderStars(width=args.width, resolution=args.resolution, center_mode=args.center_mode)
    step.bind(dataset)

    print(f"{'n_stars':>10} {'mean [s]':>9} {'min [s]':>9} {'us/star':>8}")
    for n_stars in args.n_stars:
        times = benchmark(n_stars, args.repeat, step, dataset)
        print(f"{n_stars:>10} {times.mean():>9.3f} {times.min():>9.3f} {1e6 * times.min() / n_stars:>8.2f}", flush=True)


if __name__ == "__main__":
    main()

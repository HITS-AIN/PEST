"""Compare reading the stars of TNG subhalos directly with loading them with pynbody.

`PynbodyDataset.stars` and `PynbodyDataset.particles` (used by RenderStars and
LoadParticles) read the particles of a subhalo directly from the snapshot
files, instead of loading the whole subhalo with pynbody
(`PynbodyDataset.halo`). For a few subhalos in a stellar mass range, this times
both ways. It renders both sets of stars centered on the catalogue center and
prints the largest difference of the images, which should be at float
precision, and checks that both give the same star particles.

It also compares RenderStars with how it rendered before: the whole subhalo
loaded with pynbody and centered on all its particles, including velocities.
The center shift is the distance between that center and the new one, refined
on the stars only.

The snapshot must be IllustrisTNG-like (Arepo HDF5 with a SUBFIND catalogue):

    uv run python benchmarks/tng_stars.py /path/to/TNG50-1/snapdir_099/snap_099
"""

import argparse
import time

import numpy as np
import pynbody

from pest import FilterRange, PynbodyDataset, RenderStars


def timed(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - start


def render(sim, args) -> np.ndarray:
    """Render with pynbody, in the image layout of RenderStars (top-down, channels first)."""
    image = pynbody.plot.stars.render(sim, width=args.width, resolution=args.resolution, noplot=True, return_image=True)
    return np.moveaxis(np.asarray(image[::-1], dtype=np.float32), -1, 0)


def render_before(dataset, subhalo_id, args):
    """RenderStars before reading the stars directly."""
    halo = dataset.halo(subhalo_id)
    pynbody.analysis.center(halo, mode=args.center_mode)
    return halo, render(halo, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot_path", help="e.g. .../TNG50-1/snapdir_099/snap_099")
    parser.add_argument("--min-mass", type=float, default=5.0e10, help="minimum stellar mass [Msun]")
    parser.add_argument("--max-mass", type=float, default=5.2e10, help="maximum stellar mass [Msun]")
    parser.add_argument("--count", type=int, default=3, help="number of subhalos")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--width", type=float, default=50.0, help="image width [kpc]")
    parser.add_argument("--center-mode", default="ssc", help='"ssc", "com" or "hyb"')
    parser.add_argument("--threads", type=int, default=1, help="pynbody threads, as in one pipeline worker")
    args = parser.parse_args()

    pynbody.config["number_of_threads"] = args.threads
    dataset, seconds = timed(PynbodyDataset, args.snapshot_path)
    print(f"Opened the snapshot and catalogue in {seconds:.1f}s")
    stellar_mass = FilterRange("SubhaloMassType", index=4, min=args.min_mass, max=args.max_mass, units="Msol")
    subhalo_ids = dataset.halo_numbers[stellar_mass.mask(dataset.table())][: args.count]
    if len(subhalo_ids) == 0:
        raise SystemExit("No subhalos in the stellar mass range.")
    _, seconds = timed(dataset.stars, subhalo_ids[0])
    if dataset._direct_groups.get(pynbody.family.star) is None:
        raise SystemExit("The stars of this snapshot cannot be read directly.")
    print(f"First call of stars(), including the setup: {seconds:.2f}s")
    step = RenderStars(width=args.width, resolution=args.resolution, center_mode=args.center_mode)
    step.bind(dataset)
    fields = ["pos", "vel", "mass"]

    for subhalo_id in subhalo_ids:
        stars, t_stars = timed(dataset.stars, subhalo_id)
        particles, t_particles = timed(dataset.particles, subhalo_id, "stars", fields)
        dataset.direct_loading = False
        loaded, t_halo = timed(dataset.stars, subhalo_id)
        loaded_particles, t_halo_particles = timed(dataset.particles, subhalo_id, "stars", fields)
        dataset.direct_loading = True
        loading_diff = np.abs(render(stars, args) - render(loaded, args)).max()
        same = all(np.array_equal(particles[field], loaded_particles[field]) for field in fields)
        print(f"\nSubhalo {subhalo_id} ({len(stars)} stars)")
        print(
            f"  stars():     halo() {t_halo:.2f}s, direct {t_stars:.2f}s, largest image difference {loading_diff:.1e}"
        )
        print(f"  particles(): halo() {t_halo_particles:.2f}s, direct {t_particles:.2f}s, same {fields}: {same}")

        (halo, before), t_before = timed(render_before, dataset, subhalo_id, args)
        record, t_after = timed(step, {"subhalo_id": subhalo_id})
        # The same star, relative to the catalogue center and to the old center.
        old_center = np.asarray(stars["pos"][0]) - np.asarray(halo.st["pos"][0])
        new_center = pynbody.analysis.center(
            stars, mode=args.center_mode, with_velocity=False, cen_num_particles=0, return_cen=True
        )
        shift = np.linalg.norm(old_center - np.asarray(new_center))
        render_diff = np.abs(record["image"] - before).max()
        print(
            f"  render:      before {t_before:.2f}s, after {t_after:.2f}s, "
            f"center shift {shift:.2f} kpc, largest image difference {render_diff:.1e}"
        )


if __name__ == "__main__":
    main()

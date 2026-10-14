The slow part isn't the rendering. Most of the time goes into loading each subhalo. I timed three
subhalos in your mass range (about 900k stars each, single process):

| Step | Time |
|---|---|
| `dataset.halo()` → `halos.load_copy()` | **4.0–4.8 s** |
| `pynbody.analysis.center` | 0.55–1.0 s |
| `pynbody.plot.stars.render` (256², 3 bands) | 0.8–1.2 s |

Each worker also spends about 110 s on setup (opening the snapshot and catalogue). That happens once
per worker, so it doesn't matter for long runs.

**Why loading is slow:** `load_copy` calls `pynbody.load(filename, take=...)`, which reopens the
whole 681-file TNG50 snapshot for every subhalo. Opening it alone takes about 3 s even when the
files are cached. Loading only star particles didn't help, so the time is the reopen, not the extra
gas and dark matter.

**What to change, biggest gain first:**

1. **Read star particles directly with h5py, keeping the files open.** In TNG, each subhalo's stars
   are one contiguous range in snapshot order, so you only read a slice from one or two chunk files.
   You'd read only the fields `render` needs (`Coordinates`, `Masses`, `GFM_StellarFormationTime`,
   `GFM_Metallicity`) and wrap them with `pynbody.new(star=n)`. That should bring loading from about
   4.5 s down to well under 1 s, which roughly halves or better the total per subhalo. It needs a
   new method in `PynbodyDataset` in the pest package. One catch: you'd have to handle wind
   particles (`GFM_StellarFormationTime < 0`) and the formation-time-to-age conversion yourself.
2. **Make centring cheaper:**
   - `with_velocity=False`: `render` doesn't rotate the image, so velocity centring is wasted.
     Measured, this cut centring from 1.0 s to 0.6 s on the gas-rich halo.
   - Or skip `analysis.center` and subtract `SubhaloPos` from the catalogue, with periodic wrapping.
     `_particle_field` already does this. Centring then costs nothing, at the price of centring on
     the most-bound particle instead of the stellar density peak.
3. **Avoid thread oversubscription.** pynbody defaults to `number_of_threads: -1`, so each of your
   16 workers starts OpenMP and I/O threads for every core it sees. Set
   `pynbody.config['number_of_threads'] = 1` (or cores ÷ workers) and `OMP_NUM_THREADS=1` in the
   worker initializer or `run-ice-deep.sh`, then raise `num_workers` to match `--cpus-per-task`. I
   didn't measure this one (my thread test crashed), so treat it as likely rather than confirmed.

Items 2 and 3 are a few lines each. Item 1 is the real fix. Want me to implement it in
`PynbodyDataset`/`RenderStars` and check that the images match the current output?
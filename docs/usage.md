# Usage

PEST builds training datasets with an ETL pipeline that is configured in a YAML file. The
{ref}`ETL pipeline <usage-etl>` section describes the pipeline and its configuration, and the
following sections show its three main uses: {ref}`SKIRT images <usage-skirt>`,
{ref}`point clouds <usage-pointcloud>` and {ref}`rendered star images <usage-render>`. The
{ref}`download script <usage-download>` fetches the SKIRT images.

(usage-etl)=
## ETL pipeline

PEST follows a classic **Extract → Transform → Load** pattern driven by a YAML configuration file.

```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│   Extract   │────▶│      Transform       │────▶│    Load     │
│  (dataset)  │     │  (filter / augment)  │     │  (parquet)  │
└─────────────┘     └──────────────────────┘     └─────────────┘
```

Run a pipeline from the command line:

```bash
pest pipelines/illustris_skirt.yaml
```

Extraction and transformation are fused into a single pass: each worker loads
one record and immediately runs the whole transform chain on it, instead of
rewriting the full dataset to disk once per transformation step.

### Configuration file

The configuration file sets up the parallel execution and the three stages. Each stage names a
class with `class_path` and passes its keyword arguments in `init_args`:

```yaml
num_workers: 4          # parallel workers
threads_per_worker: 1   # threads of each worker, e.g. for pynbody (default: CPUs / num_workers)
batch_size: 16          # records handed to each worker per chunk
shuffle: true           # shuffle before transformations
seed: 42

extract:
  class_path: pest.IllustrisSkirtDataset
  init_args:
    path: data/illustris_skirt
    columns: [image, simulation, snapshot, subhalo_id, [sersic_n, r]]

transform:
  - column: image
    transformations:
      - class_path: pest.ResizeImage
        init_args:
          size: [128, 128]

load:
  - class_path: pest.ParquetWriter
    init_args:
      output_path: output/dataset.parquet
```

Any extractor or transformation can be replaced by a custom class — set `class_path` to a
fully-qualified `module.ClassName` string and PEST will import and instantiate it automatically.

### Extract

An extractor class yields one record per object (e.g. galaxy).
Built-in extractors:

| Class | Input | Use |
|---|---|---|
| `IllustrisSkirtDataset` | Directory of Illustris SKIRT images | {ref}`SKIRT images <usage-skirt>` |
| `PynbodyDataset` | Simulation snapshot + halo catalogue read with [pynbody](https://pynbody.readthedocs.io/) (e.g. IllustrisTNG, EAGLE, AHF) | {ref}`Point clouds <usage-pointcloud>`, {ref}`rendered star images <usage-render>` |

### Transform

Transformations are applied sequentially to the dataset.
Classes that set `is_filter = True` are used as filters (rows are dropped); others map the data in-place.

The `transform` list accepts two kinds of entries:

- **Column-scoped** groups `{column: image, transformations: [...]}`: each transformation
  receives and returns the value of that column.
- **Record-level** steps `{class_path: ..., init_args: ...}`: a filter receives the whole
  record and returns whether to keep it, any other step receives the record and returns the
  updated record (e.g. to add new columns).

`FilterRange` keeps records whose `column` (optionally `[..., index]`, converted to `units`) lies
within `[min, max]`. It works on any numeric column, per record or vectorized on the whole table
(see below). The other transformations are described with the data they work on:
{ref}`image transforms <usage-skirt-transforms>`, {ref}`LoadParticles <usage-load-particles>` and
{ref}`RenderStars <usage-render-options>`.

**Filtering before extraction.** If the dataset provides a `table()` of cheap per-record
columns (as `PynbodyDataset` does with the halo catalogue), the *leading* record-level
filters that implement a vectorized `mask(table)` (e.g. `FilterRange`) are applied to the
whole table before any record is extracted. Filters later in the chain run per record.
Steps that implement `bind(dataset)` get access to the worker's dataset, which lets
loader steps like `LoadParticles` read large arrays only for the records that survived.
The {ref}`point cloud <usage-pointcloud>` and {ref}`star rendering <usage-render>` examples
select subhalos this way.

### Load

Loaders persist the processed dataset. Multiple loaders can be chained; each
receives the same (in-memory, already-transformed) records.

| Class | Output |
|---|---|
| `ParquetWriter` | Apache Parquet file, written in row-group batches as records arrive |
| `HuggingFaceWriter` | Reads back a `ParquetWriter` output into a HuggingFace `datasets.Dataset`, optionally pushing it to the Hub. Requires the optional `hf` extra (`pip install astro-pest[hf]`). |

See the {doc}`API reference <api>` for the full list of built-in extractors, transformations, and loaders.

(usage-skirt)=
## SKIRT images

Build an image dataset from the synthetic SDSS images of Illustris and IllustrisTNG galaxies made
with the radiative transfer code SKIRT, optionally with morphology parameters of the galaxies.
Download the images first with the {ref}`download script <usage-download>`.

This pipeline turns the images into RGB images, drops corrupt, truncated and edge-on galaxies, and
crops, resizes and normalizes the rest:

```yaml
num_workers: 16
shuffle: True
seed: 42

extract:
  class_path: pest.IllustrisSkirtDataset
  init_args:
    path: data/illustris_skirt
    columns:
      - image
      - simulation
      - snapshot
      - subhalo_id
      - [sersic_n, r]

transform:
  - column: image
    transformations:
      - class_path: pest.CreateNormalizedRGBColors
        init_args:
          stretch: 0.9
          range: 5
          lower_limit: 0.001
          channel_combinations: [[2, 3], [1, 0], [0]]
          scalers: [0.7, 0.5, 1.3]
      - class_path: pest.FilterUnhealthyData
      - class_path: pest.FilterTruncatedGalaxies
        init_args:
          threshold: 0.4
      - class_path: pest.AlignImageHorizontally
      - class_path: pest.FilterInclinationAngle
        init_args:
          max_inclination: 60
      - class_path: pest.Crop
        init_args:
          relative_size: 0.5
      - class_path: pest.ResizeImage
        init_args:
          size: [128, 128]
      - class_path: pest.ReflectionalInvariance
      - class_path: pest.GaussianBlur
        init_args:
          kernel_size: 3
      - class_path: pest.AddCircularMask
      - class_path: pest.MinMaxNormalize
        init_args:
          feature_range: [0, 1]

load:
  - class_path: pest.ParquetWriter
    init_args:
      output_path: output/illustris_skirt.parquet
```

### Columns

The `columns` argument of `IllustrisSkirtDataset` accepts plain string columns — `image`,
`simulation`, `snapshot`, `subhalo_id` — plus `(field, band)` tuples (or two-element lists, as
parsed from YAML) that look up `field` in the morphology catalog for filter `band`.

Each `(field, band)` column is read from a `morphs_{band}.hdf5` file located in the
snapshot directory of the Illustris SKIRT directory, e.g. `TNG50/sdss/snapshot_095/morphs_r.hdf5` for
`TNG50/sdss/snapshot_095/data/broadband_90.fits`. The catalog is produced by
[statmorph](https://statmorph.readthedocs.io/) and, besides `subfind_id` (used to match
rows to `subhalo_id`), exposes fields such as:

`asymmetry`, `concentration`, `deviation`, `ellipticity_asymmetry`, `ellipticity_centroid`,
`elongation_asymmetry`, `elongation_centroid`, `flag`, `flag_sersic`, `flux_circ`,
`flux_ellip`, `gini`, `gini_m20_bulge`, `gini_m20_merger`, `intensity`, `m20`,
`multimode`, `orientation_asymmetry`, `orientation_centroid`, `outer_asymmetry`, `r20`,
`r50`, `r80`, `rhalf_circ`, `rhalf_ellip`, `rmax_circ`, `rmax_ellip`, `rpetro_circ`,
`rpetro_ellip`, `sersic_amplitude`, `sersic_ellip`, `sersic_n`, `sersic_rhalf`,
`sersic_theta`, `sersic_xc`, `sersic_yc`, `shape_asymmetry`, `sky_mean`, `sky_median`,
`sky_sigma`, `smoothness`, `sn_per_pixel`.

When a `ParquetWriter` writes the loaded data, a `(field, band)` column is flattened to
the Parquet column name `{field}_{band}` (e.g. `sersic_n_r`).

(usage-skirt-transforms)=
### Image transforms

These steps work on the image array, in a column-scoped group on `image` as in the pipeline above:

| Class | Purpose |
|---|---|
| `CreateNormalizedRGBColors` | Combine multi-channel FITS data into an RGB image |
| `FilterUnhealthyData` | Drop corrupt or blank images |
| `AlignImageHorizontally` | Rotate galaxy to a canonical orientation |
| `FilterInclinationAngle` | Remove edge-on galaxies above a max inclination |
| `FilterTruncatedGalaxies` | Drop images whose galaxy extends beyond a major-axis threshold |
| `Crop` | Crop a square region around the detected galaxy centroid |
| `ResizeImage` | Rescale to a fixed pixel size |
| `AddCircularMask` | Zero out pixels outside an inscribed circle |
| `GaussianBlur` | Apply a Gaussian blur to each channel |
| `ReflectionalInvariance` | Randomly flip images for data augmentation |
| `MinMaxNormalize` | Normalise pixel values to a fixed range |

(usage-pointcloud)=
## Point clouds

Build a dataset of particle arrays, e.g. the positions, velocities and masses of the star
particles, for each subhalo of a simulation snapshot that [pynbody](https://pynbody.readthedocs.io/)
can read, such as IllustrisTNG, EAGLE or AHF.

This pipeline selects TNG50 subhalos by stellar mass and `SubhaloFlag` and loads the positions,
velocities and masses of their star particles:

```yaml
num_workers: 16
shuffle: True
seed: 42

extract:
  class_path: pest.PynbodyDataset
  init_args:
    snapshot_path: data/TNG50-1/snapdir_099/snap_099
    columns:
      - subhalo_id
      - snapshot
      - SubhaloSFR

# Leading filters run vectorized on the halo catalogue before any record is
# extracted; particle data is only loaded for the surviving subhalos.
transform:
  - class_path: pest.FilterRange # stellar mass
    init_args:
      column: SubhaloMassType
      index: 4
      min: 5.0e+10 # [Msun]
      max: 5.2e+10 # [Msun]
      units: Msol
  - class_path: pest.FilterRange # SubhaloFlag == 1
    init_args:
      column: SubhaloFlag
      min: 1
  - class_path: pest.LoadParticles
    init_args:
      component: stars
      fields:
        - pos
        - vel
        - mass

load:
  - class_path: pest.ParquetWriter
    init_args:
      output_path: output/pynbody_tng.parquet
```

### Halo catalogue

`PynbodyDataset` yields one record per halo of the catalogue and only reads catalogue
data. It works with any snapshot and halo catalogue that pynbody can read: SUBFIND
catalogues (IllustrisTNG, Arepo, Gadget) are loaded as flat subhalo catalogues, all others
(e.g. EAGLE FoF groups, AHF, AdaptaHOP, VELOCIraptor) with pynbody's default halos. Pass
`halos_args` (keyword arguments for pynbody's `halos()`) to choose the catalogue
explicitly, e.g. `{priority: [AHFCatalogue]}`.

The `columns` argument accepts `subhalo_id` (pynbody's halo number, e.g. starting at 1 for
EAGLE), `snapshot` and any catalogue property (e.g. `SubhaloSFR`). Which properties exist
depends on the catalogue format; EAGLE and AdaptaHOP catalogues provide none, so there is
nothing to filter on before extraction. Subhalo selection (mass range, `SubhaloFlag`, ...)
is done with filters in the transform stage, and particle arrays are added by a
`LoadParticles` transform step (see below), so they are only read for selected subhalos.

(usage-load-particles)=
### Loading particles

`LoadParticles` adds the particle arrays of one `component` (`stars`, `gas` or `dm`) to the
record, one column per entry of `fields`: `pos` and `vel` relative to the subhalo center, in kpc
and km/s, and any other particle field (e.g. `mass`) as a float32 array. To load several components
into one record, add a `LoadParticles` step for each and set a column name `prefix`, e.g. `gas_`.

Particles are read with partial loading where the snapshot format supports it, otherwise
(e.g. Gadget binary, Ramses) copied out of the full snapshot. Particle positions and
velocities are relative to the catalogue center (`SubhaloPos`/`SubhaloVel` or
`sub_pos`/`sub_vel`, wrapped at the periodic box boundary), or to the center computed by
`pynbody.analysis.center` if the catalogue has none.

(usage-tng-direct)=
For IllustrisTNG-like snapshots (Arepo HDF5 with a SUBFIND catalogue), `LoadParticles` and
`RenderStars` read only the arrays they need directly from the snapshot files, without
reopening the whole snapshot for every subhalo. Arrays that pynbody derives instead of
reading them (e.g. `age`) are still loaded with pynbody.

(usage-render)=
## Rendered star images

Build an image dataset by rendering the star particles of each subhalo into a 3-color image with
[pynbody](https://pynbody.readthedocs.io/). The extraction and the subhalo selection work as for
{ref}`point clouds <usage-pointcloud>`, with a `RenderStars` step in place of `LoadParticles`:

```yaml
num_workers: 16
shuffle: True
seed: 42

extract:
  class_path: pest.PynbodyDataset
  init_args:
    snapshot_path: data/TNG50-1/snapdir_099/snap_099
    columns:
      - subhalo_id
      - snapshot
      - SubhaloSFR

transform:
  - class_path: pest.FilterRange # stellar mass
    init_args:
      column: SubhaloMassType
      index: 4
      min: 5.0e+10 # [Msun]
      max: 5.2e+10 # [Msun]
      units: Msol
  - class_path: pest.FilterRange # SubhaloFlag == 1
    init_args:
      column: SubhaloFlag
      min: 1
  - class_path: pest.RenderStars
    init_args:
      width: 50 kpc
      resolution: 128

load:
  - class_path: pest.ParquetWriter
    init_args:
      output_path: output/pynbody_tng_stars.parquet
```

(usage-render-options)=
### RenderStars options

`RenderStars` renders the image with `pynbody.plot.stars.render` and adds it as a float32 array of
shape `(3, height, width)`, channels first and rows from top to bottom. Before rendering, the
center is refined on the star particles.

| Argument | Description |
|---|---|
| `width` | Image width, as a float in kpc or a unit string, e.g. `50 kpc` (default 50) |
| `resolution` | Image size in pixels (default: pynbody's default) |
| `center_mode` | pynbody centering scheme to refine the center on the stars: `ssc` (default), `com` or `hyb`. `catalogue` keeps the catalogue center |
| `column` | Name of the added column (default `image`) |
| `render_args` | Further arguments for `pynbody.plot.stars.render`, e.g. `{dynamic_range: 3.0}` or the bands `r_band`, `g_band` and `b_band` (default `I`, `V` and `U`). Dust (`with_dust`) is not supported, as pynbody computes it from the gas particles, which are not loaded |

For IllustrisTNG-like snapshots, the star particles are read
{ref}`directly from the snapshot files <usage-tng-direct>`.

(usage-download)=
## Downloading the Illustris/TNG SKIRT dataset

The `pest-download-skirt` command downloads the Illustris/TNG SKIRT synthetic image
tarballs, extracts them, and lays out the resulting FITS files the way
`IllustrisSkirtDataset` expects (`<path>/<simulation>/...`):

```bash
pest-download-skirt data/illustris_skirt
```

It requires an Illustris/TNG API key, provided via the `ILLUSTRIS_API_KEY` environment
variable. Files that already exist in the destination simulation directory are skipped,
and each tarball is deleted after being extracted.

By default it downloads a built-in list of TNG50, TNG100, and Illustris SDSS tarballs. Pass
`--urls-file` to download a custom list instead (one URL per line):

```bash
pest-download-skirt data/illustris_skirt --urls-file my_urls.txt
```

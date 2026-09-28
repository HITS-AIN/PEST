# Usage

## ETL Pipeline

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

### Extract

An extractor class yields one record per object (e.g. galaxy).
Built-in extractors:

| Class | Input |
|---|---|
| `IllustrisSkirtDataset` | Directory of Illustris SKIRT |

#### `IllustrisSkirtDataset` columns

The `columns` argument accepts plain string columns — `image`, `simulation`, `snapshot`,
`subhalo_id` — plus `(field, band)` tuples (or two-element lists, as parsed from YAML)
that look up `field` in the morphology catalog for filter `band`.

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

### Transform

Transformations are applied sequentially to the dataset.
Classes that set `is_filter = True` are used as filters (rows are dropped); others map the data in-place.

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

### Load

Loaders persist the processed dataset. Multiple loaders can be chained; each
receives the same (in-memory, already-transformed) records.

| Class | Output |
|---|---|
| `ParquetWriter` | Apache Parquet file, written in row-group batches as records arrive |
| `HuggingFaceWriter` | Reads back a `ParquetWriter` output into a HuggingFace `datasets.Dataset`, optionally pushing it to the Hub. Requires the optional `hf` extra (`pip install astro-pest[hf]`). |

### Configuration reference

```yaml
num_workers: 4       # parallel workers
batch_size: 16       # records handed to each worker per chunk
shuffle: true        # shuffle before transformations
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

See the {doc}`API reference <api>` for the full list of built-in extractors, transformations, and loaders.

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

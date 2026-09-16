# PEST — Preprocessing Engine for Spherinator Training

PEST converts raw astrophysical simulation data into clean, structured training datasets for
[Spherinator](https://github.com/HITS-AIN/Spherinator) and [HiPSter](https://github.com/HITS-AIN/HiPSter).

## Installation

```bash
pip install astro-pest
```

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

### Extract

An extractor class yields one record per object (e.g. galaxy).
Built-in extractors:

| Class | Input |
|---|---|
| `IllustrisExtractor` | Local IllustrisTNG snapshots |
| `FitsDataset` | Directory of FITS images |

### Transform

Transformations are applied sequentially to the dataset.
Classes that set `is_filter = True` are used as filters (rows are dropped); others map the data in-place.

| Class | Purpose |
|---|---|
| `CreateNormalizedRGBColors` | Combine multi-channel FITS data into an RGB image |
| `FilterUnhealthyData` | Drop corrupt or blank images |
| `AlignImageHorizontally` | Rotate galaxy to a canonical orientation |
| `FilterInclinationAngle` | Remove edge-on galaxies above a max inclination |
| `CropQuadratic` | Crop image to a square region |
| `ResizeImage` | Rescale to a fixed pixel size |
| `ReflectionalInvariance` | Randomly flip images for data augmentation |
| `MinMaxNormalize` | Normalise pixel values to a fixed range |

### Load

Loaders persist the processed dataset.

| Class | Output |
|---|---|
| `ParquetWriter` | Apache Parquet file (HuggingFace `datasets` compatible) |

### Configuration reference

```yaml
num_workers: 4       # parallel workers
shuffle: true        # shuffle before transformations
seed: 42

extract:
  class_path: pest.FitsDataset
  init_args:
    path: data/fits
    columns: [image, simulation, snapshot, subhalo_id]

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

## Point-cloud pipelines

For particle-based outputs (point clouds), the `IllustrisExtractor` yields per-galaxy particle
tables for stars, gas, and dark matter. The result is written with `ParquetWriter` without any
image transformations:

```yaml
extract:
  class_path: pest.IllustrisExtractor
  init_args:
    simulation_path: /data/Illustris
    simulation: TNG50-1
    snapshot: 99
    objects: centrals
    component:
      - name: stars
        fields: [masses, positions, velocities, ages, metallicities]
        selector:
          type: stellar mass
          min: 5.0e+10
          max: 5.2e+10

load:
  - class_path: pest.ParquetWriter
    init_args:
      output_path: output/pointcloud.parquet
```

import argparse
import importlib

import numpy as np
import yaml
from datasets import Dataset


def _instantiate(class_path: str, init_args: dict):
    """Instantiate a class from a dotted ``module.ClassName`` string."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**init_args)


def load_records(class_path, init_args):
    dataset = _instantiate(class_path, init_args)
    yield from dataset


class Pipeline:
    def __init__(
        self,
        config: dict,
    ):
        self.config = config
        self.num_workers = config.get("num_workers", 1)
        self.shuffle = config.get("shuffle", True)
        self.seed = config.get("seed", 42)

    def run(self) -> None:
        """Run the pipeline: extract, transform, and load data."""

        # Extract
        extract_cfg = self.config["extract"]
        ds = Dataset.from_generator(
            load_records,
            gen_kwargs={
                "class_path": extract_cfg["class_path"],
                "init_args": extract_cfg.get("init_args", {}),
            },
            num_proc=self.num_workers,
        )

        # Shuffle before transformations to ensure randomness in filtering and augmentation
        if self.shuffle:
            ds = ds.shuffle(seed=self.seed)

        # Transform
        transform_cfgs = self.config.get("transform", [])
        for column_cfs in transform_cfgs:
            if column_cfs["column"] != "image":
                raise NotImplementedError("Currently only 'image' column transformations are supported.")

            for transform_cfg in column_cfs.get("transformations", []):
                transform = _instantiate(transform_cfg["class_path"], transform_cfg.get("init_args", {}))

                if getattr(transform, "is_filter", False):
                    ds = ds.filter(transform, batched=False, num_proc=self.num_workers)
                else:

                    def apply(batch, t=transform):
                        images = []
                        for i, img in enumerate(batch["image"]):
                            try:
                                images.append(t(np.array(img)))
                            except Exception as e:
                                print(
                                    f"Transform {t.__class__.__name__} failed for "
                                    f"simulation={batch.get('simulation', [None])[i]}, "
                                    f"snapshot={batch.get('snapshot', [None])[i]}, "
                                    f"subhalo_id={batch.get('subhalo_id', [None])[i]}: {e}"
                                )
                                raise
                        batch["image"] = images
                        return batch

                    ds = ds.map(apply, batched=True, num_proc=self.num_workers)

        # Load
        load_cfgs = self.config.get("load", [])
        loads = [_instantiate(cfg["class_path"], cfg.get("init_args", {})) for cfg in load_cfgs]

        for load in loads:
            load(ds)


def main() -> None:
    """CLI entry point: read a YAML config file and run the pipeline."""
    parser = argparse.ArgumentParser(
        prog="pest",
        description="Preprocessing Engine for Spherinator Training",
    )
    parser.add_argument("config", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    Pipeline(config).run()


if __name__ == "__main__":
    main()

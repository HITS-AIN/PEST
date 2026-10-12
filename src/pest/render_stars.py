import numpy as np
import pynbody


class RenderStars:
    """Transform step that adds a 3-color stellar image of a subhalo to the record.

    The subhalo is centered with `pynbody.analysis.center` and rendered with
    `pynbody.plot.stars.render`. The pipeline binds the worker's dataset to
    this step, which must provide `halo(subhalo_id)` returning a snapshot of
    the subhalo particles only (e.g. `PynbodyDataset`).

    Args:
        width (float | str): Image width, as a float in the snapshot units
            (kpc for `PynbodyDataset`) or a unit string, e.g. "50 kpc".
        resolution (int | None): Image size in pixels. If `None`, the pynbody
            default is used.
        center_mode (str | None): pynbody centering scheme, e.g. "ssc", "com",
            "pot" or "hyb". If `None`, the pynbody default is used.
        column (str): Name of the added column.
        render_args (dict | None): Additional arguments for
            `pynbody.plot.stars.render`, e.g. `{"dynamic_range": 3.0}`.
    """

    def __init__(
        self,
        width: float | str = 50,
        resolution: int | None = None,
        center_mode: str | None = None,
        column: str = "image",
        render_args: dict | None = None,
    ):
        self.width = width
        self.resolution = resolution
        self.center_mode = center_mode
        self.column = column
        self.render_args = render_args or {}
        self.dataset = None

    def bind(self, dataset) -> None:
        if not hasattr(dataset, "halo"):
            raise TypeError(f"{type(dataset).__name__} does not provide halo objects.")
        self.dataset = dataset

    def __call__(self, record: dict) -> dict:
        if self.dataset is None:
            raise RuntimeError("RenderStars must be bound to a dataset before use.")
        if "subhalo_id" not in record:
            raise KeyError("RenderStars requires the 'subhalo_id' column in the extracted records.")
        # The halo is a standalone copy, so centering it does not affect the dataset.
        halo = self.dataset.halo(record["subhalo_id"])
        # pynbody fails with an obscure KeyError for a missing star family, e.g. in dark matter only runs.
        if len(halo.st) == 0:
            raise ValueError(f"Subhalo {record['subhalo_id']} has no star particles to render.")
        pynbody.analysis.center(halo, mode=self.center_mode)
        image = pynbody.plot.stars.render(
            halo,
            width=self.width,
            resolution=self.resolution,
            noplot=True,
            return_image=True,
            **self.render_args,
        )
        # pynbody puts the lowest y in the first row; flip to the usual top-down image layout.
        image = np.asarray(image[::-1], dtype=np.float32)
        # Channels first: (3, height, width).
        record[self.column] = np.moveaxis(image, -1, 0)
        return record

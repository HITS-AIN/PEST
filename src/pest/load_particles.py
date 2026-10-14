class LoadParticles:
    """Transform step that adds particle arrays of a subhalo to the record.

    Placing it after the filters in the transform chain means the (large)
    particle arrays are only read for records that survived the filters.
    The pipeline binds the worker's dataset to this step, which must provide
    `particles(subhalo_id, component, fields)` (e.g. `PynbodyDataset`).

    Args:
        fields (list[str]): Particle fields to load, e.g. `["pos", "vel", "mass"]`.
        component (str): Particle family: "stars", "gas" or "dm".
        prefix (str): Prefix for the added column names, e.g. `"gas_"` to load
            several components into one record.
    """

    def __init__(self, fields: list[str], component: str = "stars", prefix: str = ""):
        self.fields = fields
        self.component = component
        self.prefix = prefix
        self.dataset = None

    def bind(self, dataset) -> None:
        if not hasattr(dataset, "particles"):
            raise TypeError(f"{type(dataset).__name__} does not provide particle data.")
        self.dataset = dataset

    def __call__(self, record: dict) -> dict:
        if self.dataset is None:
            raise RuntimeError("LoadParticles must be bound to a dataset before use.")
        if "subhalo_id" not in record:
            raise KeyError("LoadParticles requires the 'subhalo_id' column in the extracted records.")
        particles = self.dataset.particles(record["subhalo_id"], self.component, self.fields)
        for field, values in particles.items():
            record[self.prefix + field] = values
        return record

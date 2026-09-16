def batch_generator(iterable, batch_size=100):
    """Group generator updates into chunks for better network efficiency."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

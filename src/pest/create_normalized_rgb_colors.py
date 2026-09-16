import numpy as np


class CreateNormalizedRGBColors:
    def __init__(
        self,
        stretch: float = 0.9,
        range: int = 5,
        lower_limit: float = 0.001,
        channel_combinations: list[list[int]] | None = None,
        scalers: list[float] | None = None,
    ):
        """
        Initialize CreateNormalizedRGBColors.

        Args:
            stretch (float): Stretch factor for the normalization.
            range (int): Range for the normalization.
            lower_limit (float): Lower limit for pixel values.
            channel_combinations (list[list[int]]): List of channel combinations to create RGB images.
            scalers (list[float]): List of scalers for each channel combination.
        """
        if scalers is None:
            scalers = [0.7, 0.5, 1.3]
        if channel_combinations is None:
            channel_combinations = [[2, 3], [1, 0], [0]]
        self.stretch = stretch
        self.range = range
        self.lower_limit = lower_limit
        self.channel_combinations = channel_combinations
        self.scalers = scalers

    def __call__(self, images) -> np.ndarray:
        resulting_image = np.zeros(
            (
                len(self.channel_combinations),
                images.shape[1],
                images.shape[2],
            )
        )
        for i, channel_combination in enumerate(self.channel_combinations):
            resulting_image[i] = images[channel_combination[0]]
            for t in range(1, len(channel_combination)):
                resulting_image[i] = resulting_image[i] + images[channel_combination[t]]
            resulting_image[i] = resulting_image[i] * self.scalers[i]

        mean = np.mean(resulting_image, axis=0)
        resulting_image = (
            resulting_image * np.asinh(self.stretch * self.range * (mean - self.lower_limit)) / self.range / mean
        )

        resulting_image = np.nan_to_num(resulting_image, nan=0, posinf=0, neginf=0)
        resulting_image = np.clip(resulting_image, 0, 1)
        return resulting_image

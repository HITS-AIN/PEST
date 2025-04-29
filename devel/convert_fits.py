from pest import FitsConverter

FitsConverter(
    image_size=128,
).convert_all(
    "/home/doserbd/data/two-images/TNG100/sdss/snapnum_099/data",
    "./output",
)

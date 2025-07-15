from pest import FitsConverter

FitsConverter(
    image_size=128,
    datatype="float32",
    flatten=False,
).convert_all(
    "/home/doserbd/git/SPACE_HPC_Visualization_Workshop/data/illustris/fits/TNG100/sdss/snapnum_099/data",
    "./output",
)

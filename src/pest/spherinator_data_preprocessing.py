# ruff: noqa

"""
First version of Spherinator data preprocessing routine to produce
images, 2D maps, datacubes, or point clouds of arbitrary quantities
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from scipy.stats import binned_statistic_2d, binned_statistic_dd
from skimage import img_as_ubyte
from skimage.filters import gaussian
from skimage.io import imsave
from skimage.util import img_as_float

### common functions


def map_bands_to_values(bands):
    band_mapping = {"U": 0, "B": 1, "V": 2, "K": 3, "g": 4, "r": 5, "i": 6, "z": 7}
    return [band_mapping[band] for band in bands]


def sdss_to_rgb(g, r, i, min=0, f="asinh", alpha=0.02, Q=8.0):
    I = (g + r + i) / 3.0
    if f == "asinh":
        B = g * np.arcsinh(alpha * Q * (I - min)) / Q
        G = r * np.arcsinh(alpha * Q * (I - min)) / Q
        R = i * np.arcsinh(alpha * Q * (I - min)) / Q
    elif f == "log":
        B = np.log10(g)
        G = np.log10(r)
        R = np.log10(i)
    # remove nans
    B = np.nan_to_num(B)
    G = np.nan_to_num(G)
    R = np.nan_to_num(R)
    # normalize to 0-1
    all_channels = np.stack([R, G, B])
    norm = np.max(all_channels)
    R /= norm
    G /= norm
    B /= norm

    return [R, G, B]


def rotate_galaxy(particles, orientation, spin_aperture):  # [kpc]
    if orientation == "original":
        return particles

    if orientation in ["face-on", "edge-on"]:
        rad = np.linalg.norm(particles["Coordinates"])
        inner_mask = rad < spin_aperture
        print(f"  particles within spin aperture: {sum(inner_mask)}")
        pos_inner = particles["Coordinates"][inner_mask][:, 0:3]
        vel_inner = particles["Velocities"][inner_mask][:, 0:3]
        mass_inner = particles["Masses"][inner_mask]
        sL = np.cross(pos_inner, vel_inner)
        Lvec = np.sum(mass_inner[:, np.newaxis] * sL, axis=0)
        spin = Lvec / np.linalg.norm(Lvec)
        # print('  angular momentum:', Lvec)
        print("  spin (unit) vector:", spin)

    if orientation == "random":
        # random vector as spin
        spin = np.random.normal(loc=0, scale=1, size=3)
        spin = spin / np.linalg.norm(spin)
        print("  using random orientation vector:", spin)

    pos = particles["Coordinates"][:, 0:3]
    vel = particles["Velocities"][:, 0:3]

    pos_rot = rotate_z(pos, np.arctan2(spin[0], spin[1]) * 180.0 / np.pi)
    vel_rot = rotate_z(vel, np.arctan2(spin[0], spin[1]) * 180.0 / np.pi)
    norm = rotate_z(np.array([spin]), np.arctan2(spin[0], spin[1]) * 180.0 / np.pi)[0]

    pos_rot = rotate_x(pos_rot, np.arctan2(norm[1], norm[2]) * 180.0 / np.pi)
    vel_rot = rotate_x(vel_rot, np.arctan2(norm[1], norm[2]) * 180.0 / np.pi)
    norm = rotate_x(np.array([norm]), np.arctan2(norm[1], norm[2]) * 180.0 / np.pi)[0]

    print(f"  spin in new rotated frame (should be [0,0,1]): {norm[0]:.3f},{norm[1]:.3f},{norm[2]:.3f}")

    particles["Coordinates"] = pos_rot
    particles["Velocities"] = vel_rot

    return particles


def create_2Dmap(
    particles,
    operations,
    fov,
    fov_unit,
    image_depth,
    image_scale,
    image_size,
    smoothing,
    channels,
    subid,
    component,
    orientation,
    output_path,
    output_format,
    debug,
):
    if fov_unit == "kpc":
        max_rad = fov / 2.0

    elif fov_unit == "r50":
        rad = np.linalg.norm(particles["Coordinates"], axis=1)
        if debug:
            print(
                np.min(particles["Coordinates"][:, 0]),
                np.max(particles["Coordinates"][:, 0]),
            )
            print(
                np.min(particles["Coordinates"][:, 1]),
                np.max(particles["Coordinates"][:, 1]),
            )
            print(
                np.min(particles["Coordinates"][:, 2]),
                np.max(particles["Coordinates"][:, 2]),
            )
        max_rad = (fov / 2.0) * np.percentile(rad, 50)
        print(f" min, median, max radius: {np.min(rad):.1f},{np.median(rad):.1f},{np.max(rad):.1f} kpc")

    print(f" FOV: {2 * max_rad:.1f} kpc")

    if orientation in ["face-on", "original", "random"]:
        indy = 1

    elif orientation == "edge-on":
        indy = 2

    img_x = particles["Coordinates"][:, 0]
    img_y = particles["Coordinates"][:, indy]
    # if field == "HI mass":
    #    quantity = particles["Masses"] * particles["NeutralHydrogenAbundance"]

    # define image resolution and physical extent
    nPixels = [image_size, image_size]
    minMax = [-max_rad, max_rad]  # [kpc], relative to the galaxy center
    pixelScale = 2 * max_rad / float(image_size)

    # count the number of particles on the grid
    grid_npart, _, _, _ = binned_statistic_2d(
        img_x,
        img_y,
        particles[channels[0]],
        statistic="count",
        bins=nPixels,
        range=[minMax, minMax],
    )
    print(f" particles in FOV: {np.sum(grid_npart):.0f}")

    # calculate 2D maps by projecting particles on a grid
    grid_quants = []
    for i in range(len(channels)):
        print(channels[i], operations[i])
        gq, _, _, _ = binned_statistic_2d(
            img_x,
            img_y,
            particles[channels[i]],
            statistic=operations[i],
            bins=nPixels,
            range=[minMax, minMax],
        )
        if debug:
            plt.figure()
            plt.imshow(gq, cmap="viridis")
        # set max image depth (for density maps)
        if operations[i] == "sum":
            part_mass = np.mean(particles["Masses"])
            gq = np.clip(gq, image_depth * part_mass, np.inf)
        else:
            gq[grid_npart < image_depth] = np.nan
        # scale and normalize
        if image_scale[i] == "log":
            gq = np.log10(gq)
        if debug:
            print(f" pixel value range: {np.nanmin(gq.flatten()):.2e} - {np.nanmax(gq.flatten()):.2e}")
        if np.nanmax(gq) > np.nanmin(gq):
            gq = (gq - np.nanmin(gq)) / (np.nanmax(gq) - np.nanmin(gq))
        # remove NaNs
        #    gq = np.nan_to_num(gq)
        gq = np.clip(gq, 0, 1)
        if debug:
            plt.figure()
            plt.imshow(gq)
        # collect channel arrays
        grid_quants.append(gq)
        del gq

    # Stack arrays along last dimension to produce single or multi-channel image
    image_array = np.stack((grid_quants), axis=-1)
    if debug:
        print(f" image shape: {image_array.shape}")
        print(
            f" normalized pixel value range: {np.nanmin(image_array.flatten()):.2e} - {np.nanmax(image_array.flatten()):.2e}"
        )
        plt.figure()
        plt.hist(image_array.flatten(), bins=100, color="gray", alpha=0.7)
        plt.title("Histogram of pixel values")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

    assert len(channels) == 1 or len(channels) == 3 or len(channels) == 4, (
        f"Number of channels must be one (grayscale), three (RGB) or four (RGBA)"
    )

    if len(channels) == 1:  # Grayscale
        image_array = np.squeeze(image_array)

    image = Image.fromarray((255 * image_array).astype(np.uint8))

    # apply Gaussian smoothing
    if smoothing != "None":
        image = image.filter(ImageFilter.GaussianBlur(radius=smoothing / pixelScale))

    # filepath = output_path / Path(sim, str(snapshot))
    filepath = Path(output_path)
    filepath.mkdir(parents=True, exist_ok=True)
    filename = filepath / Path(str(subid) + "_" + component + "." + output_format)
    image.save(filename)

    return


def create_opticalimage(
    particles,
    fov,
    fov_unit,
    image_size,
    smoothing,
    bands,
    scale,
    stretch,
    Q,
    subid,
    orientation,
    output_path,
    output_format,
    debug,
):
    bands_ind = map_bands_to_values(bands)

    if fov_unit == "kpc":
        max_rad = fov / 2.0

    elif fov_unit == "r50":
        rad = np.linalg.norm(particles["Coordinates"], axis=1)
        if debug:
            print(
                np.min(particles["Coordinates"][:, 0]),
                np.max(particles["Coordinates"][:, 0]),
            )
            print(
                np.min(particles["Coordinates"][:, 1]),
                np.max(particles["Coordinates"][:, 1]),
            )
            print(
                np.min(particles["Coordinates"][:, 2]),
                np.max(particles["Coordinates"][:, 2]),
            )
        max_rad = (fov / 2.0) * np.percentile(rad, 50)
        print(f" min, median, max radius: {np.min(rad):.1f},{np.median(rad):.1f},{np.max(rad):.1f} kpc")

    print(f" FOV: {2 * max_rad:.1f} kpc")

    if orientation in ["face-on", "original", "random"]:
        indy = 1

    elif orientation == "edge-on":
        indy = 2

    img_x = particles["Coordinates"][:, 0]
    img_y = particles["Coordinates"][:, indy]
    # if field == "HI mass":
    #    quantity = particles["Masses"] * particles["NeutralHydrogenAbundance"]

    # define image resolution and physical extent
    nPixels = [image_size, image_size]
    minMax = [-max_rad, max_rad]  # [kpc], relative to the galaxy center
    pixelScale = 2 * max_rad / float(image_size)

    fluxes = 10 ** (-0.4 * particles["GFM_StellarPhotometrics"])
    if debug:
        print(" flux array shape, min, max:", fluxes.shape, fluxes.min(), fluxes.max())

    # count the number of particles on the grid
    grid_npart, _, _, _ = binned_statistic_2d(
        img_x,
        img_y,
        particles["Masses"],
        statistic="count",
        bins=nPixels,
        range=[minMax, minMax],
    )
    print(f" particles in FOV: {np.sum(grid_npart):.0f}")

    # calculate 2D maps by projecting particles on a grid
    grid_values = []
    for i in range(len(bands_ind)):
        gq, _, _, _ = binned_statistic_2d(
            img_x,
            img_y,
            fluxes[:, bands_ind[i]],
            statistic="sum",
            bins=nPixels,
            range=[minMax, minMax],
        )
        if debug:
            plt.figure(figsize=(4, 4))
            plt.imshow(-2.5 * np.log10(gq))
            plt.colorbar()
            plt.title(f"input {bands[i]} band [mag]")
        if debug:
            print(f" pixel value range: {np.nanmin(gq.flatten()):.2e} - {np.nanmax(gq.flatten()):.2e}")
        # collect channel arrays
        grid_values.append(gq)
        del gq

    grid_values = sdss_to_rgb(grid_values[0], grid_values[1], grid_values[2], f=scale, alpha=stretch, Q=Q)
    if debug:
        for i, color in enumerate(["R", "G", "B"]):
            plt.figure(figsize=(4, 4))
            plt.imshow(grid_values[i])
            plt.colorbar()
            plt.title(f"scaled {color} image")

    # Stack arrays along last dimension to produce single or multi-channel image
    image_array = np.stack((grid_values), axis=-1)
    if debug:
        print(f" image shape: {image_array.shape}")
        print(
            f" normalized pixel value range: {np.nanmin(image_array.flatten()):.2e} - {np.nanmax(image_array.flatten()):.2e}"
        )
        plt.figure(figsize=(3, 2))
        plt.hist(image_array.flatten(), bins=100, color="gray", alpha=0.7)
        plt.title("Histogram of pixel values")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

    assert len(bands) == 1 or len(bands) == 3, f"Number of channels must be one (grayscale) or three (RGB)"

    if len(bands) == 1:  # Grayscale
        image_array = np.squeeze(image_array)

    image = Image.fromarray((255 * image_array).astype(np.uint8))

    # apply Gaussian smoothing
    if smoothing != "None":
        image = image.filter(ImageFilter.GaussianBlur(radius=smoothing))

    if debug:
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title("output RGB image")

    # filepath = output_path / Path(sim, str(snapshot))
    filepath = Path(output_path)
    filepath.mkdir(parents=True, exist_ok=True)
    filename = filepath / Path(str(subid) + "_" + "optical" + "." + output_format)
    image.save(filename)

    return


def create_cube_PPP(
    particles,
    operation,
    fov,
    fov_unit,
    cube_depth,
    cube_scale,
    cube_size,
    smoothing,
    field,
    subid,
    component,
    orientation,
    output_path,
    output_format,
    debug,
):
    if fov_unit == "kpc":
        max_rad = fov / 2.0

    elif fov_unit == "r90":
        rad = np.linalg.norm(particles["Coordinates"], axis=1)
        if debug:
            print(
                np.min(particles["Coordinates"][:, 0]),
                np.max(particles["Coordinates"][:, 0]),
            )
            print(
                np.min(particles["Coordinates"][:, 1]),
                np.max(particles["Coordinates"][:, 1]),
            )
            print(
                np.min(particles["Coordinates"][:, 2]),
                np.max(particles["Coordinates"][:, 2]),
            )
        max_rad = (fov / 2.0) * np.percentile(rad, 90)
        print(f" min, median, max radius: {np.min(rad):.1f},{np.median(rad):.1f},{np.max(rad):.1f} kpc")

    print(f" FOV: {2 * max_rad:.1f} kpc")

    if orientation in ["face-on", "original", "random"]:
        cube_x = particles["Coordinates"][:, 0]
        cube_y = particles["Coordinates"][:, 1]
        cube_z = particles["Coordinates"][:, 2]
    elif orientation == "edge-on":
        cube_x = particles["Coordinates"][:, 0]
        cube_y = particles["Coordinates"][:, 2]
        cube_z = -particles["Coordinates"][:, 1]

    # if field == "HI mass":
    #    quantity = particles["Masses"] * particles["NeutralHydrogenAbundance"]

    # define cube resolution and physical extent
    nPixels = [cube_size, cube_size, cube_size]
    minMax = [-max_rad, max_rad]  # [kpc], relative to the galaxy center
    pixelScale = 2 * max_rad / float(cube_size)

    # count the number of particles on the grid
    grid_npart, _, _, _ = binned_statistic_dd(
        cube_x,
        cube_y,
        cube_z,
        particles[field],
        statistic="count",
        bins=nPixels,
        range=[minMax, minMax, minMax],
    )
    print(f" particles in FOV: {np.sum(grid_npart):.0f}")

    # calculate 3D map by projecting particles on a grid
    gq, _, _, _ = binned_statistic_dd(
        img_x,
        img_y,
        particles[field],
        statistic=operation,
        bins=nPixels,
        range=[minMax, minMax],
    )
    if debug:
        plt.figure()
        plt.imshow(np.sum(gq, axis=-1), cmap="viridis")
    # set max image depth (for density maps)
    if operation == "sum":
        part_mass = np.mean(particles["Masses"])
        gq = np.clip(gq, cube_depth * part_mass, np.inf)
    else:
        gq[grid_npart < cube_depth] = np.nan
    # scale and normalize
    if cube_scale == "log":
        gq = np.log10(gq)
    if debug:
        print(f" voxel value range: {np.nanmin(gq.flatten()):.2e} - {np.nanmax(gq.flatten()):.2e}")
    if np.nanmax(gq) > np.nanmin(gq):
        gq = (gq - np.nanmin(gq)) / (np.nanmax(gq) - np.nanmin(gq))
    # remove NaNs
    #    gq = np.nan_to_num(gq)
    if debug:
        plt.figure()
        plt.imshow(np.sum(gq, axis=-1), cmap="viridis")

    # Stack arrays along last dimension to produce single or multi-channel image
    cube_array = gq
    if debug:
        print(f" image shape: {cube_array.shape}")
        print(
            f" normalized pixel value range: {np.nanmin(cube_array.flatten()):.2e} - {np.nanmax(cube_array.flatten()):.2e}"
        )
        plt.figure()
        plt.hist(cube_array.flatten(), bins=100, color="gray", alpha=0.7)
        plt.title("Histogram of pixel values")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

    # Clip the array values to [0, 1] and convert to 8-bit unsigned integer
    cube = img_as_ubyte(np.clip(cube_array, 0, 1))

    # Apply Gaussian smoothing if needed
    if smoothing != "None":
        # Convert to float for scikit-image's gaussian function
        cube = img_as_float(cube)
        sigma = smoothing / pixelScale
        # Apply Gaussian filter to 3D data
        cube = gaussian(cube, sigma=sigma, multichannel=False)
        # Convert back to 8-bit unsigned integer
        cube = img_as_ubyte(cube)

    filepath = Path(output_path)
    filepath.mkdir(parents=True, exist_ok=True)
    filename = filepath / Path(str(subid) + "_" + component)
    # save as numpy file
    if output_format == "npy":
        np.save(filename + ".npy", cube)
    # save as TIFF stack
    if output_format == "tiff":
        # Convert to float for scikit-image's imsave function
        cube = img_as_float(cube)
        imsave(filename + ".tiff", cube, compression="lzw")
    size = os.path.getsize(filename)
    print(f" saved PPP cube to {filename} with size {size / 1024:.3f} KB")

    return


def create_cube_PPV(
    particles,
    operation,
    fov,
    fov_unit,
    cube_depth,
    cube_scale,
    cube_size,
    smoothing,
    field,
    subid,
    component,
    orientation,
    output_path,
    output_format,
    debug,
):
    if fov_unit == "kpc":
        max_rad = fov / 2.0

    elif fov_unit == "r90":
        rad = np.linalg.norm(particles["Coordinates"], axis=1)
        vrad = np.linalg.norm(particles["Velocities"], axis=1)
        if debug:
            print(
                np.min(particles["Coordinates"][:, 0]),
                np.max(particles["Coordinates"][:, 0]),
            )
            print(
                np.min(particles["Coordinates"][:, 1]),
                np.max(particles["Coordinates"][:, 1]),
            )
            print(
                np.min(particles["Velocities"][:, 2]),
                np.max(particles["Velocities"][:, 2]),
            )
        max_rad = (fov / 2.0) * np.percentile(rad, 90)
        max_vrad = (fov / 2.0) * np.percentile(vrad, 90)
        print(f" min, median, max radius: {np.min(rad):.1f},{np.median(rad):.1f},{np.max(rad):.1f} kpc")
        print(f" min, median, max v_rad: {np.min(vrad):.1f},{np.median(vrad):.1f},{np.max(vrad):.1f} kpc")

    print(f" FOV: {2 * max_rad:.1f} kpc")

    if orientation in ["face-on", "original", "random"]:
        cube_x = particles["Coordinates"][:, 0]
        cube_y = particles["Coordinates"][:, 1]
        cube_z = particles["Velocities"][:, 2]
    elif orientation == "edge-on":
        cube_x = particles["Coordinates"][:, 0]
        cube_y = particles["Coordinates"][:, 2]
        cube_z = -particles["Velocities"][:, 1]

    # if field == "HI mass":
    #    quantity = particles["Masses"] * particles["NeutralHydrogenAbundance"]

    # define cube resolution and physical extent
    nPixels = [cube_size, cube_size, cube_size]
    minMax = [-max_rad, max_rad]  # [kpc], relative to the galaxy center
    minMaxV = [-max_vrad, max_vrad]
    # pixelScale = 2 * max_rad / float(cube_size)

    # count the number of particles on the grid
    grid_npart, _, _, _ = binned_statistic_dd(
        cube_x,
        cube_y,
        cube_z,
        particles[field],
        statistic="count",
        bins=nPixels,
        range=[minMax, minMax, minMaxV],
    )
    print(f" particles in FOV: {np.sum(grid_npart):.0f}")

    # calculate 3D map by projecting particles on a grid
    gq, _, _, _ = binned_statistic_dd(
        cube_x,
        cube_y,
        cube_z,
        particles[field],
        statistic=operation,
        bins=nPixels,
        range=[minMax, minMax, minMaxV],
    )
    if debug:
        plt.figure()
        plt.imshow(np.sum(gq, axis=-1), cmap="viridis")
    # set max image depth (for density maps)
    if operation == "sum":
        part_mass = np.mean(particles["Masses"])
        gq = np.clip(gq, cube_depth * part_mass, np.inf)
    else:
        gq[grid_npart < cube_depth] = np.nan
    # scale and normalize
    if cube_scale == "log":
        gq = np.log10(gq)
    if debug:
        print(f" voxel value range: {np.nanmin(gq.flatten()):.2e} - {np.nanmax(gq.flatten()):.2e}")
    if np.nanmax(gq) > np.nanmin(gq):
        gq = (gq - np.nanmin(gq)) / (np.nanmax(gq) - np.nanmin(gq))
    # remove NaNs
    #    gq = np.nan_to_num(gq)
    if debug:
        plt.figure()
        plt.imshow(np.sum(gq, axis=-1), cmap="viridis")

    cube_array = gq
    if debug:
        print(f" image shape: {cube_array.shape}")
        print(
            f" normalized pixel value range: {np.nanmin(cube_array.flatten()):.2e} - {np.nanmax(cube_array.flatten()):.2e}"
        )
        plt.figure()
        plt.hist(cube_array.flatten(), bins=100, color="gray", alpha=0.7)
        plt.title("Histogram of pixel values")
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")

    # Clip the array values to [0, 1] and convert to 8-bit unsigned integer
    cube = img_as_ubyte(np.clip(cube_array, 0, 1))

    # Apply Gaussian smoothing
    if smoothing != "None":
        # Convert to float for scikit-image's gaussian function
        cube = img_as_float(cube)
        sigma = smoothing / pixelScale
        # Apply Gaussian filter to 3D data
        cube = gaussian(cube, sigma=sigma, multichannel=False)
        # Convert back to 8-bit unsigned integer
        cube = img_as_ubyte(cube)

    filepath = Path(output_path)
    filepath.mkdir(parents=True, exist_ok=True)
    filename = filepath / Path(str(subid) + "_" + component)
    # save as numpy file
    if output_format == "npy":
        np.save(filename + ".npy", cube)
    # save as TIFF stack
    if output_format == "tiff":
        # Convert to float for scikit-image's imsave function
        cube = img_as_float(cube)
        imsave(filename + ".tiff", cube, compression="lzw")
    size = os.path.getsize(filename)
    print(f" saved PPV cube to {filename} with size {size / 1024:.3f} KB")

    return


def rotate_x(ar, angle):
    """Rotates the snapshot about the current x-axis by 'angle' degrees."""
    angle *= np.pi / 180
    mat = np.matrix(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )
    return np.array(np.dot(mat, ar.transpose()).transpose())


def rotate_y(ar, angle):
    """Rotates the snapshot about the current y-axis by 'angle' degrees."""
    angle *= np.pi / 180
    mat = np.matrix(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    return np.array(np.dot(mat, ar.transpose()).transpose())


def rotate_z(ar, angle):
    """Rotates the snapshot about the current z-axis by 'angle' degrees."""
    angle *= np.pi / 180
    mat = np.matrix(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return np.array(np.dot(mat, ar.transpose()).transpose())

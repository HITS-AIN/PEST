#!/usr/bin/env python
# coding: utf-8

# Many flake8 errors to be fixed
# flake8: noqa

"""
First version of Spherinator data preprocessing routine to produce
images, 2D maps, datacubes, or point clouds of arbitrary quantities
"""

import os
from pathlib import Path

import h5py
import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
import requests
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


### API helper functions


def get_illustris_api_key():
    """Return the API key for the IllustrisTNG API.
    The key can be set as an environment variable or in a file."""
    if "ILLUSTRIS_API_KEY" in os.environ:
        return os.environ["ILLUSTRIS_API_KEY"]
    elif os.path.isfile("illustris_api_key.txt"):
        with open("illustris_api_key.txt", "r") as file:
            return file.read().rstrip()
    raise ValueError(
        "No API key found. Please set the ILLUSTRIS_API_KEY environment variable"
        "or create a file named 'illustris_api_key.txt' with the API key"
    )


def get(path, key, params=None):
    # make HTTP GET request to path
    headers = {"api-key": key}
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers["content-type"] == "application/json":
        return r.json()  # parse json responses automatically

    if "content-disposition" in r.headers:
        filename = r.headers["content-disposition"].split("filename=")[1]
        with open(filename, "wb") as f:
            f.write(r.content)
        return filename  # return the filename string

    return r


### Main routine to read data using API
def data_preprocess_api(
    sim="TNG100-1",
    snapshot=99,
    objects="centrals",
    selection_type="stellar mass",
    min_mass=1e8,
    max_mass=np.inf,
    component="stars",
    output_type="optical image",
    operations=["sum"],
    fov=1.5,
    fov_unit="kpc",
    depth=4,  # [particles]
    size=128,
    smoothing="None",  # [kpc]
    channels=["Masses"],
    bands=["u", "g", "r"],
    scale="log",
    stretch=0.02,
    Q=8.0,
    orientation="original",
    spin_aperture=30.0,  # [kpc]
    catalog_fields=["SubhaloStarMetallicity", "SubhaloSFR"],
    resolution_limit=1e9,  # [Msun]
    output_path="./images/",
    output_format="png",
    debug=False,
):
    """Preprocess data using IllustrisTNG API.

    Args:
        sim (str): Name of the simulation. Default is "TNG100-1".
        snapshot (int): Snapshot number. Default is 99.
        objects (str): Type of objects to process. Default is "centrals".
        selection_type (str): Type of selection. Default is "stellar mass".
        min_mass (float): Minimum mass for selection. Default is 1e8.
        max_mass (float): Maximum mass for selection. Default is np.inf.
        component (str): Component to process. Default is "stars".
        output_type (str): Type of output. Default is "optical image".
        operation (str): Operation to perform. Default is "sum".
        fov (float): Field of view in kpc. Default is 'scaled'.
        fov_unit (str): Unit of field of view. Default is 'kpc'.
        depth (int): Image depth in particles. Default is 4.
        size (int): Image size. Default is 128.
        smoothing (float): Smoothing radius in pixels. Default is 'None'.
        channels (int): List of image channels. Default is ["Masses"].
        scale (str): Image scaling. Default is "log".
        orientation (str): Orientation of the image. Default is "face-on".
        spin_aperture (float): Aperture for spin calculation in kpc. Default is 30.0.
        catalog_fields (list): List of fields to output in catalog. Default is ["SubhaloStarMetallicity", "SubhaloSFR"].
        resolution_limit (float): Resolution limit in Msun. Default is 1e9.
        output_path (str): Output path for images. Default is "./images/".
        debug (bool): Debug mode flag. Default is False.

    Returns:
        dict: Processed catalog data.
    """

    print(
        f"Parameters:\n"
        f" simulation: {sim}\n"
        f" snapshot: {snapshot}\n"
        f" objects: {objects}\n"
        f" selection_type: {selection_type}\n"
        f" min_mass: {min_mass:.2e} Ms, max_mass: {max_mass:.2e} Ms\n"
        f" component: {component}\n"
        f" output_type: {output_type}\n"
        f" operations: {operations}\n"
        f" fov: {fov}\n"
        f" fov_unit: {fov_unit}\n"
        f" depth: {depth} particles\n"
        f" size: {size} pixels\n"
        f" smoothing: {smoothing} pixels\n"
        f" channels: {channels}\n"
        f" bands: {bands}\n"
        f" scale: {scale}\n"
        f" Q: {Q}\n"
        f" range: {range}\n"
        f" orientation: {orientation}\n"
        f" spin_aperture: {spin_aperture:.1f} kpc\n"
        f" catalog_fields: {catalog_fields}\n"
        f" resolution_limit: {resolution_limit:.2e} Ms\n"
        f" output_path: {output_path}\n"
    )

    global mass_units_msun, dist_units_kpc

    assert output_type in [
        "optical image",
        "2D map",
        "PPP cube",
        "PPV cube",
        "none",
    ], f"{output_type} is not a valid output type."

    if objects == "centrals":
        primary_flag = [1]
    if objects == "satellites":
        primary_flag = [0]
    if objects == "all":
        primary_flag = [0, 1]

    if component == "stars":
        ptype = 4
    if component == "gas":
        ptype = 0
    if component == "dm":
        ptype = 1

    # if field == "Masses":
    #     comp_list = "Coordinates,Velocities,Masses"
    # else:
    #     comp_list = "'Coordinates,Velocities,Masses," + field + "'"
    # print("particle fields:", comp_list)

    # define API url
    url = "http://www.tng-project.org/api/" + sim + "/"

    # get api key
    api_key = get_illustris_api_key()

    # get header info
    sim_info = get(url, api_key)
    print(sim_info)
    box_size = sim_info["boxsize"] / sim_info["hubble"]
    print(f"Box size:", box_size / 1e3, " Mpc")
    mass_units_msun = 1e10 / sim_info["hubble"]
    dist_units_kpc = 1.0 / sim_info["hubble"]

    if selection_type == "total mass":
        sorting = "'-mass'"
    if selection_type == "stellar mass":
        sorting = "'-mass_stars'"
    print(f"\nSorting halos by {sorting}")
    subhalos = get(
        url + "snapshots/" + str(snapshot) + "/subhalos/",
        api_key,
        {"limit": 10000, "order_by": sorting},
    )
    print(f"Number of subhalos in catalog: {subhalos['count']}\n")

    print(f"selecting only {objects} with {selection_type} > {resolution_limit:.2e} Ms")
    print(f" and within {selection_type} range {min_mass:.2e} < M/Ms < {max_mass:.2e}")

    # form the search_query string
    search_query = (
        "?mass_stars__gt=" + str(min_mass / mass_units_msun) + "&mass_stars__lt=" + str(max_mass / mass_units_msun)
    )
    subhalos = get(
        url + "snapshots/" + str(snapshot) + "/subhalos/" + search_query,
        api_key,
        {"order_by": sorting},
    )
    print(f"\nNumber of subhalos in mass range: {subhalos['count']}\n")
    print(type(subhalos["count"]))

    # Loop over subhalos to read galaxy properties and particle data
    subid = []
    groupid = []
    m_stars = []
    m_halo = []
    m_tot = []
    r_half = []
    var0 = []
    var1 = []

    for i in range(subhalos["count"]):
        if debug:
            print(subhalos["results"][i]["url"])

        subhalo = get(subhalos["results"][i]["url"], api_key)
        mass_stars = subhalo["mass_stars"] * mass_units_msun
        mass_tot = subhalo["mass"] * mass_units_msun

        # sub_details = get(subhalo['meta']['url']+'info.json')
        sub_details = get(subhalo["meta"]["info"], api_key)
        v0 = sub_details[catalog_fields[0]]  # raw simulation units
        v1 = sub_details[catalog_fields[1]]

        group = get(subhalo["related"]["parent_halo"] + "info.json", api_key)
        mass_halo = group["Group_M_Crit200"] * mass_units_msun
        rvir_halo = group["Group_R_Crit200"] * dist_units_kpc

        if selection_type == "stellar mass":
            mass = mass_stars
        if selection_type == "total mass":
            mass = mass_tot

        if subhalo["subhaloflag"] != 1:
            continue
        if mass > max_mass:
            continue
        if mass < min_mass or mass < resolution_limit:
            break
        if subhalo["primary_flag"] not in primary_flag:
            continue

        subid.append(subhalo["id"])
        groupid.append(subhalo["grnr"])
        m_stars.append(mass_stars)
        m_tot.append(mass_tot)
        m_halo.append(mass_halo)
        r_half.append(subhalo["halfmassrad_stars"] * dist_units_kpc)
        var0.append(v0)
        var1.append(v1)

        # print galaxy info
        print("\n Galaxy:", i, " groupID:", groupid[-1], " subID:", subid[-1])
        print(f" Mstars = {m_stars[-1]:.2e} Ms, Rhalf = {r_half[-1]:.2f} kpc, Mtot = {mass_tot:.2e} Ms")

        # load galaxy particles
        # cutout = get(subhalo["cutouts"]["subhalo"], api_key, {component: comp_list})  # to load only specific components
        cutout = get(subhalo["cutouts"]["subhalo"], api_key)

        if debug:
            print(f" Npart:{subhalo['len_stars']}")

        subhalo_pos = np.array([subhalo["pos_x"], subhalo["pos_y"], subhalo["pos_z"]]) * dist_units_kpc
        subhalo_vel = np.array([subhalo["vel_x"], subhalo["vel_y"], subhalo["vel_z"]])
        if debug:
            print(f" subhalo position: {subhalo_pos[0]:.2f},{subhalo_pos[1]:.2f},{subhalo_pos[2]:.2f}")

        particles = {}
        with h5py.File(cutout, "r") as f:
            # Iterate through the items in the file and convert to dictionary
            for key in f["PartType" + str(ptype)].keys():
                data = f["PartType" + str(ptype)][key][()]
                particles[key] = data

        # delete cutout file
        os.remove(cutout)

        # center coordinates and adjust for periodic boundaries
        adjusted_coordinates = particles["Coordinates"] * dist_units_kpc - subhalo_pos
        adjusted_coordinates = np.mod(adjusted_coordinates + box_size / 2.0, box_size) - box_size / 2.0
        particles["Coordinates"] = adjusted_coordinates
        # center velocities
        particles["Velocities"] = particles["Velocities"] - subhalo_vel
        particles["Masses"] = particles["Masses"] * mass_units_msun

        print(f" number of {component} particles: {len(particles['Masses'])}")
        print(f" total particle mass: {np.sum(particles['Masses']):.1e} Ms")

        # rotate galaxy to desired orientation based on disk spin
        particles = rotate_galaxy(particles, orientation, spin_aperture)

        # create and save 2D map
        if output_type == "2D map":
            print(" creating 2D map...")
            create_2Dmap(
                particles,
                operations,
                fov,
                fov_unit,
                depth,
                scale,
                size,
                smoothing,
                channels,
                subid[-1],
                component,
                orientation,
                output_path,
                output_format,
                debug,
            )

        # create and save optical image
        if output_type == "optical image":
            print(" creating image...")
            create_opticalimage(
                particles,
                fov,
                fov_unit,
                size,
                smoothing,
                bands,
                scale,
                stretch,
                Q,
                subid[-1],
                orientation,
                output_path,
                output_format,
                debug,
            )

        # create and save PPP cube
        if output_type == "PPP cube":
            print(" creating PPP cube...")
            create_cube_PPP(
                particles,
                operations,
                fov,
                depth,
                scale,
                size,
                smoothing,
                channels,
                subid[-1],
                component,
                orientation,
                output_path,
                debug,
            )

        if output_type == "PPV cube":
            print(" creating PPV cube...")
            create_cube_PPV(
                particles,
                operations,
                fov,
                depth,
                scale,
                size,
                smoothing,
                channels,
                subid[-1],
                component,
                orientation,
                output_path,
                debug,
            )

        if output_type == "point cloud":
            print(" creating point cloud...")
            pointcloud = [
                particles["Coordinates"],
                particles["Velocities"],
                particles[field],
            ]

    print("\nCreating catalog...")
    catalog_props = [
        "SubID",
        "GroupID",
        "logMstar",
        "logMtot",
        "logMhalo",
        "Rhalf",
    ] + catalog_fields
    print(" properties:", catalog_props)
    array_list = [
        subid,
        groupid,
        np.log10(m_stars),
        np.log10(m_tot),
        np.log10(m_halo),
        r_half,
        var0,
        var1,
    ]
    catalog = {}
    for prop_name, array in zip(catalog_props, array_list):
        catalog[str(prop_name)] = array

    print("... done")

    return catalog


# To test routine run:
#
# result = data_preprocess_api(
#                 sim='TNG50-2',
#                 selection_type='stellar mass',
#                 min_mass=5e10, max_mass=5.2e10, # [Msun]
#                 component='stars',
#                 objects='centrals',
#                 field='Masses',
#                 fov='scaled', # [kpc]
#                 image_depth=1., #  1 particles per pixel (min. S/N=sqrt(depth))
#                 image_size=128,
#                 smoothing=0.0,  # [kpc]
#                 image_scale='log',
#                 orientation='original',
#                 output_path='./images_test_api/',
#                 debug=False)
#
# result should be exactly:
#
# {'SubID': [79417, 79580, 79811, 83918, 86024],
# 'GroupID': [99, 100, 102, 131, 149],
# 'logMstar': array([10.70847058, 10.710352  , 10.7067737 , 10.70629613, 10.6992251 ]),
# 'logMtot': array([12.27970075, 12.2991249 , 12.42821564, 12.25034364, 12.11282093]),
# 'logMhalo': array([12.22113751, 12.31243984, 12.3679076 , 12.17308883, 12.03844093]),
# 'Rhalf': [2.9622084440507828,
#  5.76645999409507,
#  3.437998228520815,
#  4.131237082964276,
#  5.301299084735755],
# 'SubhaloStarMetallicity': [0.024265184998512268,
#  0.022633448243141174,
#  0.02354022115468979,
#  0.024240750819444656,
#  0.023321228101849556],
# 'SubhaloSFR': [0.0,
#  0.32513657212257385,
#  0.0,
#  0.04936574399471283,
#  4.49891996383667]}


### Read data using local files
def data_preprocess_local(
    sim="TNG100-1",
    snapshot=99,
    filepath="./sims.TNG/",
    objects="centrals",
    selection_type="stellar mass",
    min_mass=1e8,
    max_mass=np.inf,
    component="stars",
    output_type="2D projection",
    field="Masses",
    operation="sum",
    fov=None,  # [kpc]
    image_depth=1,  # [particles]
    image_size=128,
    smoothing="None",  # [kpc]
    channels=1,
    image_scale="log",
    orientation="face-on",
    spin_aperture=30.0,  # [kpc]
    catalog_fields=["SubhaloStarMetallicity", "SubhaloSFR"],
    resolution_limit=1e9,  # [Msun]
    output_path="./images/",
    debug=False,
):
    """Preprocess local simulation data.

    Args:
        sim (str): Name of the simulation.
        snapshot (int): Snapshot number.
        filepath (str): Path to the simulation files.
        objects (str): Type of objects to consider (centrals, satellites, all).
        selection_type (str): Type of selection (stellar mass, total mass).
        min_mass (float): Minimum mass for selection.
        max_mass (float): Maximum mass for selection.
        component (str): Component to analyze (stars, gas, dm).
        output_type (str): Type of output (2D projection, point cloud).
        field (str): Field to analyze.
        operation (str): Operation to perform.
        fov (float): Field of view in kpc.
        image_depth (int): Image depth in particles.
        image_size (int): Image size.
        smoothing (float): Smoothing factor in kpc.
        channels (int): Number of channels.
        image_scale (str): Image scaling method.
        orientation (str): Orientation of the image.
        spin_aperture (float): Spin aperture in kpc.
        catalog_fields (list): List of catalog fields.
        resolution_limit (float): Resolution limit in Msun.
        output_path (str): Path to save the output images.
        debug (bool): Enable debug mode.

    Returns:
        dict: Catalog containing selected properties for each object.
    """

    print(
        f"Parameters:\n"
        f" simulation: {sim}\n"
        f" snapshot: {snapshot}\n"
        f" filepath: {filepath}\n"
        f" objects: {objects}\n"
        f" selection_type: {selection_type}\n"
        f" min_mass: {min_mass:.2e} Ms, max_mass: {max_mass:.2e} Ms\n"
        f" component: {component}\n"
        f" output_type: {output_type}\n"
        f" field: {field}\n"
        f" operation: {operation}\n"
        f" fov: {fov}\n"
        f" image_depth: {image_depth} particles\n"
        f" image_size: {image_size} pixels\n"
        f" smoothing: {smoothing} kpc\n"
        f" channels: {channels}\n"
        f" image_scale: {image_scale}\n"
        f" orientation: {orientation}\n"
        f" spin_aperture: {spin_aperture:.1f} kpc\n"
        f" catalog_fields: {catalog_fields}\n"
        f" resolution_limit: {resolution_limit:.2e} Ms\n"
        f" output_path: {output_path}\n"
    )

    global mass_units_msun, dist_units_kpc

    if component == "stars":
        ptype = 4
    if component == "gas":
        ptype = 0
    if component == "dm":
        ptype = 1

    if objects == "centrals":
        primary_flag = [1]
    if objects == "satellites":
        primary_flag = [0]
    if objects == "all":
        primary_flag = [0, 1]

    # define path to data
    basePath = filepath + sim + "/output/"

    # define conversions to physical units
    header = il.groupcat.loadHeader(basePath, 99)
    mass_units_msun = 1e10 / header["HubbleParam"]
    dist_units_kpc = header["Time"] / header["HubbleParam"]

    # load subhalos (i.e. galaxies)
    print(f"loading galaxy catalog from {basePath}...")
    subhalos = il.groupcat.loadSubhalos(basePath, snapshot)
    subhalos["SubhaloID"] = np.arange(subhalos["count"])

    # Loop over galaxies to read galaxy properties and particle data
    sub_id = []
    group_id = []
    m_stellar = []
    m_halo = []
    m_tot = []
    r_half = []
    var0 = []
    var1 = []
    print(f"selecting only {objects} with {selection_type} > {resolution_limit:.2e} Ms")
    print(f" and within {selection_type} range {min_mass:.2e} < M/Ms < {max_mass:.2e}")

    # select mass range
    m_stars_all = subhalos["SubhaloMassType"][:, 4] * mass_units_msun
    m_tot_all = subhalos["SubhaloMass"] * mass_units_msun
    if selection_type == "stellar mass":
        mass_mask = (m_stars_all > min_mass) * (m_stars_all < max_mass)
    if selection_type == "total mass":
        mass_mask = (m_tot_all > min_mass) * (m_tot_all < max_mass)
    print(f" ... selected {sum(mass_mask)} subhalos in mass range")

    i = 0
    for subid in subhalos["SubhaloID"][mass_mask]:
        # subhalo properties
        subhalo = il.groupcat.loadSingle(basePath, snapshot, subhaloID=subid)
        ms = subhalo["SubhaloMassType"][4] * mass_units_msun
        mtot = subhalo["SubhaloMass"] * mass_units_msun
        rh = subhalo["SubhaloHalfmassRadType"][4] * dist_units_kpc
        v0 = subhalo[catalog_fields[0]]  # raw simulation units
        v1 = subhalo[catalog_fields[1]]

        # Group properties
        gid = subhalo["SubhaloGrNr"]
        group = il.groupcat.loadSingle(basePath, snapshot, haloID=gid)
        mh = group["Group_M_Crit200"] * mass_units_msun
        central_subid = group["GroupFirstSub"]
        if central_subid == subid:
            central_flag = 1
        else:
            central_flag = 0

        if selection_type == "stellar mass":
            mass = ms
        if selection_type == "total mass":
            mass = mtot

        if debug and subid % 1000 == 0:
            print(
                "\nGalaxy:",
                i,
                " subif:",
                subid,
                " gid:",
                gid,
                "flag:",
                central_flag,
                "mass:",
                np.log10(mass),
            )

        if subhalo["SubhaloFlag"] != 1:
            continue
        if mass < resolution_limit:
            continue
        if mass < min_mass:
            continue
        if mass > max_mass:
            continue
        if central_flag not in primary_flag:
            continue

        # print galaxy info
        print(
            "\nGalaxy:",
            i,
            " subID:",
            subid,
            " groupID:",
            gid,
            " primary_flag:",
            central_flag,
        )
        print(f" Mstars={ms:.2e} Ms, Rhalf={rh:.2f} kpc, Mhalo={mh:.2e} Ms")

        # load galaxy particles
        print(" loading particles...")
        particles = il.snapshot.loadSubhalo(basePath, snapshot, subid, component)  # all fields
        # print number of particles in the galaxy
        masses_temp = particles["Masses"] * mass_units_msun
        print(f" number of {component} particles: {len(masses_temp)}")
        print(f" total particle mass: {np.sum(masses_temp):.1e} Ms")

        # center the coordinates/velocities and masses
        particles["Coordinates"] = (particles["Coordinates"] - subhalo["SubhaloPos"]) * dist_units_kpc
        particles["Velocities"] = particles["Velocities"] - subhalo["SubhaloVel"]
        particles["Masses"] = particles["Masses"] * mass_units_msun

        # rotate galaxy
        particles = rotate_galaxy(particles, orientation, spin_aperture)

        # create and save image
        if output_type == "2D projection":
            print(" creating image...")
            create_image(
                particles,
                field,
                operation,
                fov,
                image_depth,
                image_scale,
                image_size,
                smoothing,
                channels,
                subid,
                component,
                orientation,
                output_path,
                debug,
            )

        if output_type == "point cloud":
            print(" creating point cloud...")
            pointcloud = {
                "Coordinates": particles["Coordinates"],
                "Velocities": particles["Velocities"],
                field: particles[field],
            }

        sub_id.append(subid)
        group_id.append(gid)
        m_stellar.append(ms)
        m_halo.append(mh)
        m_tot.append(mtot)
        r_half.append(rh)
        var0.append(v0)
        var1.append(v1)
        i += 1

        del subhalo, group

    print("\nCreating catalog...")
    catalog_props = ["SubID", "GroupID", "logMstar", "logMtot", "logMhalo", "Rhalf"]
    catalog_props = catalog_props + catalog_fields
    print(" properties:", catalog_props)
    array_list = [
        sub_id,
        group_id,
        np.log10(m_stellar),
        np.log10(m_tot),
        np.log10(m_halo),
        r_half,
        var0,
        var1,
    ]
    catalog = {}
    for prop_name, array in zip(catalog_props, array_list):
        catalog[str(prop_name)] = array

    print("... done")

    del subhalos

    return catalog


# Test:
#
# result = data_preprocess_local(
#                 sim='TNG50-1',
#                 selection_type='stellar mass',
#                 min_mass=5e10, max_mass=5.2e10, # [Msun]
#                 component='stars',
#                 objects='centrals',
#                 field='Masses',
#                 fov='scaled', # [kpc]
#                 image_depth=1., #  1 particles per pixel (min. S/N=sqrt(depth))
#                 image_size=128,
#                 smoothing=0.0,  # [kpc]
#                 image_scale='log',
#                 orientation='original',
#                 output_path='./images_test_local/',
#                 debug=False)
#
# result:
#
# {'SubID': [79417, 79580, 79811, 83918, 86024],
#  'GroupID': [99, 100, 102, 131, 149],
#  'logMstar': array([10.7084704 , 10.71035241, 10.70677389, 10.70629614, 10.69922539]),
#  'logMtot': array([12.27970013, 12.29912528, 12.42821589, 12.25034259, 12.11282079]),
#  'logMhalo': array([12.22113751, 12.31243984, 12.3679076 , 12.17308883, 12.03844093]),
#  'Rhalf': [2.9622568716719884,
#   5.76649192233405,
#   3.4379403007604177,
#   4.131219575001772,
#   5.301296235820272],
#  'SubhaloStarMetallicity': [0.024265185,
#   0.022633448,
#   0.023540221,
#   0.02424075,
#   0.023321228],
#  'SubhaloSFR': [0.0, 0.32513657, 0.0, 0.049365744, 4.49892]}

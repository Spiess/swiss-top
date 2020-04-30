import argparse
import os
from multiprocessing import Pool

import fiona
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

import stl


def create_grid(topography, spacing):
    xmax = np.max(topography[:, 0])
    xmin = np.min(topography[:, 0])
    ymax = np.max(topography[:, 1])
    ymin = np.min(topography[:, 1])

    width = xmax - xmin
    grid_width = int(width // spacing) + 1
    height = ymax - ymin
    grid_height = int(height // spacing) + 1

    grid = np.zeros((grid_width, grid_height))
    grid[:, :] = np.nan

    for x, y, z in topography:
        grid[int((x - xmin) // spacing), int((y - ymin) // spacing)] = z

    return grid, xmin, ymin


def subselect(topography, n, spacing=200):
    """Returns a primitively subsampled version of the topography by only keeping every nth point in both directions."""
    grid, xmin, ymin = create_grid(topography, spacing)

    subselection_grid = grid[::n, ::n]

    subselection = []

    for y in range(subselection_grid.shape[1]):
        for x in range(subselection_grid.shape[0]):
            if not np.isnan(subselection_grid[x, y]):
                subselection.append(((x * n * spacing) + xmin, (y * n * spacing) + ymin, subselection_grid[x, y]))

    subselection = np.array(subselection)

    return subselection


def create_mesh(topography, spacing):
    """Returns an STL mesh of the given topography."""
    grid, xmin, ymin = create_grid(topography, spacing)

    faces = []

    for y in range(grid.shape[1] - 1):
        for x in range(grid.shape[0] - 1):
            if not (np.isnan(grid[x, y]) or np.isnan(grid[x + 1, y]) or np.isnan(grid[x, y + 1])):
                faces.append(np.array(
                    [
                        [x * spacing + xmin, (y + 1) * spacing + ymin, grid[x, y + 1]],
                        [(x + 1) * spacing + xmin, y * spacing + ymin, grid[x + 1, y]],
                        [x * spacing + xmin, y * spacing + ymin, grid[x, y]]
                    ]
                ))
            if not (np.isnan(grid[x + 1, y + 1]) or np.isnan(grid[x + 1, y]) or np.isnan(grid[x, y + 1])):
                faces.append(np.array(
                    [
                        [(x + 1) * spacing + xmin, (y + 1) * spacing + ymin, grid[x + 1, y + 1]],
                        [x * spacing + xmin, (y + 1) * spacing + ymin, grid[x, y + 1]],
                        [(x + 1) * spacing + xmin, y * spacing + ymin, grid[x + 1, y]]
                    ]
                ))

    mesh = stl.mesh.Mesh(np.zeros(len(faces), dtype=stl.mesh.Mesh.dtype))
    mesh.vectors = np.array(faces)

    return mesh


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('topography_file', help='Path to topography file in .xyz format (e.g. DHM200.xyz).')
    parser.add_argument('border_file', help='Path to border file in shapefile format (e.g. '
                                            'swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET.shp).')
    parser.add_argument('output_directory', help='Path to output directory.')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--subselection-factor', type=int, default=1)

    args = parser.parse_args()

    topography_file = args.topography_file
    border_file = args.border_file
    output_dir = args.output_directory
    threads = args.threads
    subselection_factor = args.subselection_factor

    print('Reading border...')
    with fiona.open(border_file) as container:
        border = np.array(list(container)[0]['geometry']['coordinates'][0])

    print('Reading topography...')
    topography = np.load(topography_file) if topography_file.endswith('.npy') else np.loadtxt(topography_file)
    if subselection_factor > 1:
        topography = subselect(topography, subselection_factor)
    topography2d = topography[:, :2]

    border2d = border[:, :2]

    border_path = Path(border2d)

    print('Checking borders...')
    chunks = np.array_split(topography2d, threads, axis=0)

    with Pool(threads) as p:
        completed_chunks = p.map(border_path.contains_points, chunks)

    contained_points = np.concatenate(completed_chunks)

    final_topography = topography[contained_points, :]

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'topography.npy'), final_topography)

    print('Plotting results...')
    plt.figure(figsize=(15, 10))
    plt.scatter(final_topography[:, 0], final_topography[:, 1], c=final_topography[:, 2], marker='.')
    plt.plot(border2d[:, 0], border2d[:, 1])
    plt.show()

    print('Creating mesh...')
    mesh = create_mesh(final_topography, 200 * subselection_factor)

    faces = mesh.vectors
    faces -= np.min(faces, axis=(0, 1))
    faces /= np.max(faces)
    offset = np.max(faces, axis=(0, 1)) / 2
    offset[2] = 0
    faces -= offset

    mesh.vectors = faces

    mesh.save(os.path.join(output_dir, 'topography.stl'))


if __name__ == '__main__':
    main()

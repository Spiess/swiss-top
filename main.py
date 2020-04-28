import argparse
import os
from multiprocessing import Pool

import fiona
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

import stl


def create_grid(topology, spacing):
    xmax = np.max(topology[:, 0])
    xmin = np.min(topology[:, 0])
    ymax = np.max(topology[:, 1])
    ymin = np.min(topology[:, 1])

    width = xmax - xmin
    grid_width = int(width // spacing) + 1
    height = ymax - ymin
    grid_height = int(height // spacing) + 1

    grid = np.zeros((grid_width, grid_height))
    grid[:, :] = np.nan

    for x, y, z in topology:
        grid[int((x - xmin) // spacing), int((y - ymin) // spacing)] = z

    return grid, xmin, ymin


def subselect(topology, n, spacing=200):
    """Returns a primitively subsampled version of the topology by only keeping every nth point in both directions."""
    grid, xmin, ymin = create_grid(topology, spacing)

    subselection_grid = grid[::n, ::n]

    subselection = []

    for y in range(subselection_grid.shape[1]):
        for x in range(subselection_grid.shape[0]):
            if not np.isnan(subselection_grid[x, y]):
                subselection.append(((x * n * spacing) + xmin, (y * n * spacing) + ymin, subselection_grid[x, y]))

    subselection = np.array(subselection)

    return subselection


def create_mesh(topology, spacing):
    """Returns an STL mesh of the given topology."""
    grid, xmin, ymin = create_grid(topology, spacing)

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

    parser.add_argument('topology_file', help='Path to topology file in .xyz format (e.g. DHM200.xyz).')
    parser.add_argument('border_file', help='Path to border file in shapefile format (e.g. '
                                            'swissBOUNDARIES3D_1_3_TLM_LANDESGEBIET.shp).')
    parser.add_argument('output_directory', help='Path to output directory.')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--subselection-factor', type=int, default=1)

    args = parser.parse_args()

    topology_file = args.topology_file
    border_file = args.border_file
    output_dir = args.output_directory
    threads = args.threads
    subselection_factor = args.subselection_factor

    print('Reading border...')
    with fiona.open(border_file) as container:
        border = np.array(list(container)[0]['geometry']['coordinates'][0])

    print('Reading topology...')
    topology = np.load(topology_file) if topology_file.endswith('.npy') else np.loadtxt(topology_file)
    if subselection_factor > 1:
        topology = subselect(topology, subselection_factor)
    topology2d = topology[:, :2]

    border2d = border[:, :2]

    border_path = Path(border2d)

    print('Checking borders...')
    chunks = np.array_split(topology2d, threads, axis=0)

    with Pool(threads) as p:
        completed_chunks = p.map(border_path.contains_points, chunks)

    contained_points = np.concatenate(completed_chunks)

    final_topology = topology[contained_points, :]

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'topology.npy'), final_topology)

    print('Plotting results...')
    plt.figure(figsize=(15, 10))
    plt.scatter(final_topology[:, 0], final_topology[:, 1], c=final_topology[:, 2], marker='.')
    plt.plot(border2d[:, 0], border2d[:, 1])
    plt.show()

    print('Creating mesh...')
    mesh = create_mesh(final_topology, 200 * subselection_factor)

    faces = mesh.vectors
    faces -= np.min(faces, axis=(0, 1))
    faces /= np.max(faces)
    offset = np.max(faces, axis=(0, 1)) / 2
    offset[2] = 0
    faces -= offset

    mesh.vectors = faces

    mesh.save(os.path.join(output_dir, 'topology.stl'))


if __name__ == '__main__':
    main()

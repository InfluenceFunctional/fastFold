import numpy as np
import torch
import tqdm
from ase import Atoms
from ase.visualize import view
import plotly.graph_objects as go
from torch_scatter import scatter
import torch.nn.functional as F


chain_length = 20
num_beads = chain_length * 2
base_pairs = [[0, -1],
              ]
time_steps = 10000
time_step = .01

BACKBONE_SPACING = 1.5  # 1.5
BASE_SPACING = BACKBONE_SPACING
hard_force_constant = -10
soft_force_constant = -1
repulsive_force_constant = 10

'''
initialize sequence as straight line
'''
bead_coords = np.arange(chain_length).repeat(2)[:, None] * np.array((1, 0, 0))[None, :].repeat(num_beads, axis=0) * BACKBONE_SPACING
bead_coords[1::2, 1] += BASE_SPACING
bead_coords += np.random.randn(*bead_coords.shape) * 0.1
bead_type = np.tile([6, 5], chain_length)

'''
initialize force constants
'''
attractive_forces = np.zeros((num_beads, num_beads))
for i in range(num_beads):
    if bead_type[i] == 6:  # backbone bead - bind to backbone neighbors and adjacent nucleobase
        if (i + 2) < num_beads:  # bind to backbone neighbor
            attractive_forces[i, i + 2] = hard_force_constant
        # if (i - 2) > 0:
        #     attractive_forces[i, i - 2] = hard_force_constant
        attractive_forces[i, i + 1] = hard_force_constant  # bind to nucleobase
    # if bead_type[i] == 5:  # nucleobase bead - bind to adjacent backbone
    #     attractive_forces[i, i - 1] = hard_force_constant

for pair in base_pairs:
    attractive_forces[2 * pair[0] + 1, 2 * pair[1] + 1] = soft_force_constant

attractive_forces += attractive_forces.T
attractive_forces = torch.Tensor(attractive_forces)

repulsive_forces = torch.zeros_like(attractive_forces)
for i in range(num_beads):
    for j in range(num_beads):
        if attractive_forces[i, j] == 0:
            repulsive_forces[i, j] = repulsive_force_constant

num_attractive_forces = torch.sum(torch.count_nonzero(attractive_forces))
attractive_forces_list = np.zeros(num_attractive_forces)
attractive_forces_pair_inds = np.zeros((num_attractive_forces, 2), dtype=np.int32)
ind = 0
for i in range(num_beads):
    for j in range(num_beads):
        if attractive_forces[i, j] != 0:
            attractive_forces_list[ind] = attractive_forces[i, j]
            attractive_forces_pair_inds[ind] = i, j
            ind += 1

num_repulsive_forces = torch.sum(torch.count_nonzero(repulsive_forces)) - num_beads
repulsive_forces_list = np.zeros(num_repulsive_forces)
repulsive_forces_pair_inds = np.zeros((num_repulsive_forces, 2), dtype=np.int32)
ind = 0
for i in range(num_beads):
    for j in range(num_beads):
        if repulsive_forces[i, j] != 0:
            if i != j:
                repulsive_forces_list[ind] = repulsive_forces[i, j]
                repulsive_forces_pair_inds[ind] = i, j
                ind += 1



def enforce_1d_bound(x: torch.tensor, x_span, x_center, mode='soft'):  # soft or hard
    """
    constrains function to range x_center plus/minus x_span
    Parameters
    ----------
    x
    x_span
    x_center
    mode

    Returns
    -------

    """
    if mode == 'soft':  # smoothly converge to (center-span,center+span)
        bounded = F.tanh((x - x_center) / x_span) * x_span + x_center
    elif mode == 'hard':  # linear scaling to hard stop at [center-span, center+span]
        bounded = F.hardtanh((x - x_center) / x_span) * x_span + x_center
    else:
        raise ValueError("bound must be of type 'hard' or 'soft'")

    return bounded


'''
propagate
'''
repulsive_forces_list = torch.Tensor(repulsive_forces_list)
attractive_forces_list = torch.Tensor(attractive_forces_list)

trajectory = torch.zeros((time_steps, num_beads, 3))
current_coords = torch.Tensor(bead_coords)
attractive_forces_pair_inds = torch.tensor(attractive_forces_pair_inds, dtype=torch.long)
repulsive_forces_pair_inds = torch.tensor(repulsive_forces_pair_inds, dtype=torch.long)

for ts in tqdm.tqdm(range(time_steps)):
    attractive_pair_vectors = current_coords[attractive_forces_pair_inds[:, 0]] - current_coords[attractive_forces_pair_inds[:, 1]]
    vector_norms = torch.linalg.norm(attractive_pair_vectors, dim=-1)
    attractive_force_magnitude = -torch.abs((vector_norms > BASE_SPACING) * attractive_forces_list)
    attractive_force_vectors = attractive_pair_vectors / vector_norms[:, None] * attractive_force_magnitude[:, None]

    repulsive_pair_vectors = current_coords[repulsive_forces_pair_inds[:, 0]] - current_coords[repulsive_forces_pair_inds[:, 1]]
    vector_norms = torch.linalg.norm(repulsive_pair_vectors, dim=-1)
    repulsive_force_magnitude = torch.abs((vector_norms < BACKBONE_SPACING) * repulsive_forces_list)
    repulsive_force_vectors = repulsive_pair_vectors / vector_norms[:, None] * repulsive_force_magnitude[:, None]

    current_coords += torch.randn_like(current_coords) * 0.05 + \
                      scatter(attractive_force_vectors, attractive_forces_pair_inds[:, 0], dim=0, reduce='sum') * time_step + \
                      scatter(repulsive_force_vectors, repulsive_forces_pair_inds[:, 0], dim=0, reduce='sum') * time_step

    trajectory[ts] = current_coords

print_steps = np.arange(time_steps - torch.sum(torch.isnan(trajectory[:, 0, 0])))
steps = [Atoms(symbols=bead_type, positions=trajectory[i]) for i in print_steps]
view(steps)

import plotly.express as px

fig = px.imshow(attractive_forces)
fig.show()
fig = px.imshow(repulsive_forces)
fig.show()

dists = torch.cdist(trajectory[-1, :], trajectory[-1, :])
fig = px.imshow(dists)
fig.show()
aa = 0

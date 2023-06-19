import numpy as np
import torch
import tqdm
from ase import Atoms
from ase.visualize import view
import plotly.graph_objects as go
from torch_scatter import scatter
import torch.nn.functional as F

device = 'cpu'
chain_length = 80
num_beads = chain_length * 2
base_pairs = [[0, -1],
              [1, -2],
              [2, -3],
              [3, -4],
              [22, -33],
              [23, -34],
              [24, -35],
              [60, 70],
              [61, -69],
              ]
time_steps = 100000
time_step = .001

BACKBONE_SPACING = 1.5  # 1.5
BASE_SPACING = BACKBONE_SPACING
backbone_force_constant = 100
soft_force_constant = 25
repulsive_force_constant = 500

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
LJ_forces = np.zeros((num_beads, num_beads))
for i in range(num_beads):
    if bead_type[i] == 6:  # backbone bead - bind to backbone neighbors and adjacent nucleobase
        if (i + 2) < num_beads:  # bind to backbone neighbor
            LJ_forces[i, i + 2] = backbone_force_constant
        LJ_forces[i, i + 1] = backbone_force_constant  # bind to nucleobase
LJ_forces += LJ_forces.T
LJ_forces = torch.Tensor(LJ_forces)

attractive_forces = np.zeros((num_beads, num_beads))
for pair in base_pairs:
    attractive_forces[2 * pair[0] + 1, 2 * pair[1] + 1] = soft_force_constant

attractive_forces += attractive_forces.T
attractive_forces = torch.Tensor(attractive_forces)

repulsive_forces = torch.zeros_like(attractive_forces)
for i in range(num_beads):
    for j in range(num_beads):
        if (attractive_forces[i, j]) == 0 and (LJ_forces[i, j] == 0):
            repulsive_forces[i, j] = repulsive_force_constant

num_LJ_forces = torch.sum(torch.count_nonzero(LJ_forces))
LJ_forces_list = np.zeros(num_LJ_forces)
LJ_forces_pair_inds = np.zeros((num_LJ_forces, 2), dtype=np.int32)
ind = 0
for i in range(num_beads):
    for j in range(num_beads):
        if LJ_forces[i, j] != 0:
            LJ_forces_list[ind] = LJ_forces[i, j]
            LJ_forces_pair_inds[ind] = i, j
            ind += 1

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


def LJ_pot(dists, magnitude, radius):
    d6 = torch.pow(radius / dists, 6)
    return 4 * magnitude * (d6 ** 2 - d6)


def LJ_force(dists, magnitude, radius):
    d6 = radius ** 6 / torch.pow(dists, 7)
    d12 = radius ** 12 / torch.pow(dists, 13)
    return 48 * magnitude * (d12 - 0.5 * magnitude * d6)


def HARM_force(dists, magnitude, radius):
    return - 2 * magnitude * (dists - radius)


'''
propagate
'''
repulsive_forces_list = torch.Tensor(repulsive_forces_list).to(device)
attractive_forces_list = torch.Tensor(attractive_forces_list).to(device)
LJ_forces_list = torch.Tensor(LJ_forces_list).to(device)

trajectory = torch.zeros((time_steps, num_beads, 3)).to(device)
current_coords = torch.Tensor(bead_coords).to(device)
LJ_forces_pair_inds = torch.tensor(LJ_forces_pair_inds, dtype=torch.long).to(device)
attractive_forces_pair_inds = torch.tensor(attractive_forces_pair_inds, dtype=torch.long).to(device)
repulsive_forces_pair_inds = torch.tensor(repulsive_forces_pair_inds, dtype=torch.long).to(device)

for ts in tqdm.tqdm(range(time_steps)):
    LJ_pair_vectors = current_coords[LJ_forces_pair_inds[:, 0]] - current_coords[LJ_forces_pair_inds[:, 1]]
    vector_norms = torch.linalg.norm(LJ_pair_vectors, dim=-1)
    LJ_force_magnitude = HARM_force(vector_norms, LJ_forces_list, BACKBONE_SPACING)
    LJ_force_vectors = LJ_pair_vectors / vector_norms[:, None] * LJ_force_magnitude[:, None]

    attractive_pair_vectors = current_coords[attractive_forces_pair_inds[:, 0]] - current_coords[attractive_forces_pair_inds[:, 1]]
    vector_norms = torch.linalg.norm(attractive_pair_vectors, dim=-1)
    attractive_force_magnitude = -torch.abs((vector_norms > BASE_SPACING) * attractive_forces_list)
    attractive_force_vectors = attractive_pair_vectors / vector_norms[:, None] * attractive_force_magnitude[:, None]

    repulsive_pair_vectors = current_coords[repulsive_forces_pair_inds[:, 0]] - current_coords[repulsive_forces_pair_inds[:, 1]]
    vector_norms = torch.linalg.norm(repulsive_pair_vectors, dim=-1)
    repulsive_force_magnitude = torch.abs((vector_norms < (1.2 * BACKBONE_SPACING)) * repulsive_forces_list)
    repulsive_force_vectors = repulsive_pair_vectors / vector_norms[:, None] * repulsive_force_magnitude[:, None]

    current_coords += torch.randn_like(current_coords).to(device) * 0.01 + \
                      scatter(LJ_force_vectors, LJ_forces_pair_inds[:, 0], dim=0, reduce='sum') * time_step + \
                      scatter(attractive_force_vectors, attractive_forces_pair_inds[:, 0], dim=0, reduce='sum') * time_step + \
                      scatter(repulsive_force_vectors, repulsive_forces_pair_inds[:, 0], dim=0, reduce='sum') * time_step

    trajectory[ts] = current_coords
    assert torch.sum(torch.isnan(current_coords)) == 0

trajectory = trajectory.cpu()

print_steps = np.arange(time_steps - torch.sum(torch.isnan(trajectory[:, 0, 0])))[::10]
steps = [Atoms(symbols=bead_type, positions=trajectory[i]) for i in print_steps]
view(steps)

import plotly.express as px

# fig = px.imshow(LJ_forces)
# fig.show()
# fig = px.imshow(attractive_forces)
# fig.show()
# fig = px.imshow(repulsive_forces)
# fig.show()

dists = torch.cdist(trajectory[-1, :], trajectory[-1, :])
fig = px.imshow(dists.numpy())
fig.show()

aa = 0

import numpy as np
import torch
import tqdm
from ase import Atoms
from ase.visualize import view
import plotly.graph_objects as go
import plotly.express as px
import torch.nn.functional as F
from torch_scatter import scatter
from torch import backends

from utils import initialize_force_matrix, init_chain, get_force_list, apply_force, get_dihedral

device = 'cuda'
chain_length = 20
num_beads = chain_length * 2
base_pairs = [[5, 18],
              [6, 17],
              [7, 16]
              ]
time_steps = 10000
time_step = .001

BACKBONE_SPACING = 1.5  # 1.5
BASE_SPACING = BACKBONE_SPACING
INIT_POSITIONAL_NOISE = .1
HELIX_ANGLE = 0.15

backbone_force_constant = 100
soft_force_constant = 25
repulsive_force_constant = 500
dihedral_force_constant = 50

'''
initialize sequence as straight line
'''
bead_coords, bead_type = init_chain(chain_length, num_beads, BACKBONE_SPACING, BASE_SPACING, INIT_POSITIONAL_NOISE)

'''
initialize force constants
'''
backbone_force_matrix = \
    initialize_force_matrix(num_beads, bead_type, mode='backbone', strength=backbone_force_constant)
pair_attraction_force_matrix = \
    initialize_force_matrix(num_beads, bead_type, mode='pairs', base_pairs=base_pairs, strength=soft_force_constant)
repulsive_force_matrix = \
    torch.ones((num_beads, num_beads)) * repulsive_force_constant
repulsive_force_matrix.fill_diagonal_(0)

backbone_force_list, backbone_pair_inds = \
    get_force_list(backbone_force_matrix, device)
pair_attraction_force_list, pair_attraction_pair_inds = \
    get_force_list(pair_attraction_force_matrix, device)
repulsive_force_list, repulsive_pair_inds = \
    get_force_list(repulsive_force_matrix, device)

dihedral_tuples_list = np.zeros((chain_length - 1, 4))
ind = 0
for i in range(0, num_beads - 2, 2):  # define dihedral frames from base to base
    dihedral_tuples_list[ind, :] = i+1, i, i + 2, i + 3
    ind += 1

dihedral_tuples_list = torch.tensor(dihedral_tuples_list, device=device, dtype=torch.long)

'''
propagate
'''
trajectory = torch.zeros((time_steps, num_beads, 3)).to(device)
current_coords = torch.Tensor(bead_coords).to(device)

force_types = ['backbone', 'pair attraction', 'repulsive']
forces_lists = [backbone_force_list, pair_attraction_force_list, repulsive_force_list]
pair_inds_lists = [backbone_pair_inds, pair_attraction_pair_inds, repulsive_pair_inds]

if device == 'cuda':
    backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

with torch.no_grad():
    for ts in tqdm.tqdm(range(time_steps)):
        atomwise_forces = []

        # pairwise forces
        for force_type, force_list, pair_inds in zip(force_types, forces_lists, pair_inds_lists):
            atomwise_forces.extend(apply_force(force_type, force_list, pair_inds, current_coords, BACKBONE_SPACING, BASE_SPACING))  # gather all forces

        # dihedral (helical) force
        dihedral_angles = get_dihedral(dihedral_tuples_list, current_coords)
        dihedral_force = -2*dihedral_force_constant*(dihedral_angles - HELIX_ANGLE)
        # todo compute force vector
        for i in range(len(dihedral_force)):
            dihedral_inds = dihedral_tuples_list[i]
            axial_coords = current_coords[dihedral_inds[1:2]]
            dihedral_axis = axial_coords[1]-axial_coords[0]
            arm1 = current_coords[dihedral_inds[0]] - current_coords[dihedral_inds[1]]
            arm2 = current_coords[dihedral_inds[3]] - current_coords[dihedral_inds[2]]

        current_coords += scatter(torch.stack(atomwise_forces), torch.cat(pair_inds_lists)[:, 0], dim=0, reduce='sum', dim_size=num_beads) * time_step \
                          + torch.randn_like(current_coords).to(device) * 0.01  # small amount of random jitter

        trajectory[ts] = current_coords

trajectory = trajectory.cpu()

print_steps = np.arange(time_steps - torch.sum(torch.isnan(trajectory[:, 0, 0])))[::10]
steps = [Atoms(symbols=bead_type, positions=trajectory[i]) for i in print_steps]
view(steps)

# fig = px.imshow(backbone_force_matrix)
# fig.show()
# fig = px.imshow(pair_attraction_force_matrix)
# fig.show()
# fig = px.imshow(repulsive_force_matrix)
# fig.show()

dists = torch.cdist(trajectory[-1, :], trajectory[-1, :])
fig = px.imshow(dists.numpy())
fig.show()

aa = 0

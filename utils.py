import torch
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter


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


def initialize_force_matrix(num_beads, bead_type, mode, base_pairs=None, strength=1):

    forces = np.zeros((num_beads, num_beads))
    if mode == 'backbone':
        for i in range(num_beads):
            if bead_type[i] == 6:  # backbone bead - bind to backbone neighbors and adjacent nucleobase
                if (i + 2) < num_beads:  # bind to backbone neighbor
                    forces[i, i + 2] = 1
                forces[i, i + 1] = 1  # bind to nucleobase
        forces += forces.T
        forces = torch.Tensor(forces)
    elif mode == 'pairs':
        for pair in base_pairs:
            forces[2 * pair[0] + 1, 2 * pair[1] + 1] = 1

    forces = forces + forces.T

    return torch.Tensor(forces * strength)


def init_chain(chain_length, num_beads, BACKBONE_SPACING, BASE_SPACING, noise):
    bead_coords = np.arange(chain_length).repeat(2)[:, None] * np.array((1, 0, 0))[None, :].repeat(num_beads, axis=0) * BACKBONE_SPACING
    bead_coords[1::2, 1] += BASE_SPACING
    bead_coords += np.random.randn(*bead_coords.shape) * noise
    bead_type = np.tile([6, 5], chain_length)

    return bead_coords, bead_type


def get_force_list(force_matrix: torch.tensor, device):
    num_beads = len(force_matrix)
    num_forces = torch.sum(torch.count_nonzero(force_matrix))
    force_list = torch.zeros(num_forces)
    pair_inds = np.zeros((num_forces, 2), dtype=np.int32)
    ind = 0
    for i in range(num_beads):
        for j in range(num_beads):
            if force_matrix[i, j] != 0:
                force_list[ind] = force_matrix[i, j]
                pair_inds[ind] = i, j
                ind += 1

    return torch.tensor(force_list, dtype=torch.float32, device=device), torch.tensor(pair_inds, dtype=torch.long, device=device)


def apply_force(force_type, force_list, pair_inds, current_coords, BACKBONE_SPACING, BASE_SPACING):

    pair_vectors = current_coords[pair_inds[:, 0]] - current_coords[pair_inds[:, 1]]
    vector_norms = torch.linalg.norm(pair_vectors, dim=-1)

    if force_type == 'backbone':
        force_magnitude = HARM_force(vector_norms, force_list, BACKBONE_SPACING)
    elif force_type == 'pair attraction':
        force_magnitude = -torch.abs((vector_norms > BASE_SPACING) * force_list)
    elif force_type == 'repulsive':
        force_magnitude = F.relu(-(vector_norms - (BACKBONE_SPACING))) ** 2 * force_list

    force_vectors = pair_vectors / vector_norms[:, None] * force_magnitude[:, None]

    return force_vectors


def get_dihedral(pair_index, coords):
    '''
    compute elements for radial & spherical embeddings
    '''
    coords_i = coords[pair_index[:, 0]]
    coords_j = coords[pair_index[:, 1]]
    coords_k = coords[pair_index[:, 2]]
    coords_l = coords[pair_index[:, 3]]

    # https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
    b1 = coords_j - coords_i
    b2 = coords_k - coords_j
    b3 = coords_l - coords_k

    n1 = torch.cross(b1, b2)
    n1 /= torch.linalg.norm(n1,dim=-1)[:, None]

    n2 = torch.cross(b2, b3)
    n2 /= torch.linalg.norm(n2,dim=-1)[:, None]

    m1 = torch.cross(n1, b2/torch.linalg.norm(b2,dim=-1)[:, None])
    x = (n1 * n2).sum(-1)
    y = (m1 * n2).sum(-1)

    phi = torch.atan2(y, x)

    # plane1 = torch.cross(coords_i, coords_j)
    # plane2 = torch.cross(coords_k, coords_l)
    #
    # a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    # b = torch.cross(plane1, plane2).norm(dim=-1)  # sin_angle * |plane1| * |plane2|
    # phi2 = torch.atan2(b, a)  # -pi to pi

    return phi

import numpy as onp
import jax
import jax.numpy as np
import argparse
import sys
import numpy as onp
import matplotlib.pyplot as plt
from jax.config import config
import torch

torch.manual_seed(0)

# Set numpy printing format
onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
onp.set_printoptions(precision=10)

# np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
# np.set_printoptions(precision=5)

# Manage arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_oris', type=int, default=20)
parser.add_argument('--num_grains', type=int, default=40000)
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--domain_x', type=float, help='Unit: mm', default=0.5)
parser.add_argument('--domain_y', type=float, help='Unit: mm', default=0.2)
parser.add_argument('--domain_z', type=float, help='Unit: mm', default=0.05)
parser.add_argument('--dt', type=float, help='Unit: s', default=2e-7)
parser.add_argument('--T_melt', type=float, help='Unit: K', default=1700.)
parser.add_argument('--T_ambient', type=float, help='Unit: K', default=300.)
parser.add_argument('--rho', type=float, help='Unit: kg/mm^3', default=8.08e-6)
parser.add_argument('--c_p', type=float, help='Unit: J/(kg*K)', default=770.)
# parser.add_argument('--laser_vel', type=float, help='Unit: mm/s', default=500.)
parser.add_argument('--power', type=float, help='Unit: W', default=60.)
parser.add_argument('--power_fraction', type=float, help='Unit: None', default=0.4)
parser.add_argument('--r_beam', type=float, help='Unit: mm', default=0.05)
parser.add_argument('--emissivity', type=float, help='Unit:', default=0.2)
parser.add_argument('--SB_constant', type=float, help='Unit: W/(mm^2*K^4)', default=5.67e-14)
parser.add_argument('--h_conv', type=float, help='Unit: W/(mm^2*K)', default=1e-4)
parser.add_argument('--kappa_T', type=float, help='Unit: W/(mm*K)', default=1e-2) 
parser.add_argument('--write_sol_interval', type=int, help='interval of writing solutions to file', default=500)

args = parser.parse_args()

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})



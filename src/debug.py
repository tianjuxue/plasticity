import numpy as np
import ufl
import dolfinx
import glob
import os
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, nls, log
from src.arguments import args
from src.utils import walltime


@walltime
def simulation():
    case_name = 'mechanical'
    ambient_T = args.T_ambient
    rho = args.rho
    Cp = args.c_p
    k = args.kappa_T
    h = args.h_conv
    eta = args.power_fraction
    r = args.r_beam
    P = args.power
    EPS = 1e-8

    x0 = 0.2*args.domain_x
    y0 = 0.5*args.domain_y

    total_t = 1200*1e-6
    vel = 0.6*args.domain_x/total_t
    dt = 1e-6
    ts = np.arange(0., total_t + dt, dt)
    print(f"total time steps = {len(ts)}")
    ele_size = 0.01

    Nx, Ny, Nz = round(args.domain_x/ele_size), round(args.domain_y/ele_size), round(args.domain_z/ele_size)

    print(f"Nx = {Nx}, Ny = {Ny}, Nz = {Nz}")

    domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0., 0., 0.]), np.array([args.domain_x, args.domain_y, args.domain_z])], 
                             [Nx, Ny, Nz], cell_type=mesh.CellType.tetrahedron)  # cell_type=mesh.CellType.hexahedron

    deg_stress = 2
    W0_ele = ufl.FiniteElement("Quadrature", domain.ufl_cell(), degree=deg_stress, quad_scheme='default')
    W0 = fem.FunctionSpace(domain, W0_ele)


    metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
    dxm = ufl.dx(metadata=metadata)


    v = fem.Function(W0)
    v.x.array[:] = 3.

    dv = ufl.TrialFunction(W0)
    v_ = ufl.TestFunction(W0)

    a_proj = dv*v_*dxm
    b_proj = 3.*v*v_*dxm

    problem = fem.petsc.LinearProblem(a_proj, b_proj, bcs=[], petsc_options={"ksp_type": "preonly"})
    u = problem.solve()

    print(u.x.array)


if __name__ == '__main__':
    simulation()

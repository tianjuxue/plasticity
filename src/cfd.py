import ufl
import numpy as np
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, nls, log
from src.utils import walltime


def ini_condition(x):
    center = np.array([0.25, 0.25, 0.])
    radius = 0.2
    distance = np.sum((x - center[:, None])**2, axis=0)
    return np.where(distance < radius**2, 1., 0.)


def dirichlet_boundaries(x):
    return np.logical_or(np.isclose(x[0], 0.), np.isclose(x[1], 0.))


@walltime
def simulation():
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0., 0.]), np.array([1., 1.])], [100, 100])

    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    h = np.min(dolfinx.cpp.mesh.h(domain, tdim, range(num_cells)))
 
    mesh_vtk_file = io.VTKFile(MPI.COMM_WORLD, f'data/vtk/cfd/mesh/mesh.pvd', 'w')
    mesh_vtk_file.write_mesh(domain)
    sol_vtk_file = io.VTKFile(MPI.COMM_WORLD, f'data/vtk/cfd/sols/sol.pvd', 'w')


    xdmf_file = io.XDMFFile(MPI.COMM_WORLD, f'data/xdmf/cfd/sol.xdmf', "w")
    xdmf_file.write_mesh(domain)

    P1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
    V = fem.FunctionSpace(domain, P1)

    u_pre = fem.Function(V)
    u_pre.interpolate(ini_condition)

    u_crt = fem.Function(V)
    u_crt.x.array[:] = u_pre.x.array

    xdmf_file.write_function(u_pre, 0.)

    facets = mesh.locate_entities_boundary(domain, dim=1, marker=dirichlet_boundaries)
    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bcs = [fem.dirichletbc(value=PETSc.ScalarType(0.), dofs=dofs, V=V)]

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt = 1e-3
    ts = np.arange(0., 401*dt, dt)
    
    beta = fem.Constant(domain, (PETSc.ScalarType(1.), PETSc.ScalarType(1.)))

    # v_supg = v + 0.1*h*ufl.dot(beta, ufl.grad(v))


    v_supg = v + 0.2*h*ufl.dot(beta, ufl.grad(v))

    # a = u*v_supg*ufl.dx + dt*ufl.dot(beta, ufl.grad(u))*v_supg*ufl.dx
    # L = u_pre*v_supg*ufl.dx
    # problem = fem.petsc.LinearProblem(a, L, bcs=bcs)

    F_res = (u_crt - u_pre)*v_supg*ufl.dx + dt*ufl.dot(beta, ufl.grad(u_crt))*v_supg*ufl.dx - 1e1*dt*2*u_crt*(1-u_crt)*(2*u_crt-1)*v*ufl.dx

    # F_res = (u_crt - u_pre)*v_supg*ufl.dx + dt*ufl.dot(beta, ufl.grad(u_crt))*v_supg*ufl.dx

    problem = dolfinx.fem.petsc.NonlinearProblem(F_res, u_crt, bcs)
    solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)


    for i in range(len(ts[1:])):

        # uh = problem.solve()
        # u_pre.x.array[:] = uh.x.array

        solver.solve(u_crt)
        u_pre.x.array[:] = u_crt.x.array

        if i % 20 == 0:
            print(f"At step {i + 1}")
            t = ts[i + 1]
            xdmf_file.write_function(u_pre, t)
            sol_vtk_file.write_function(u_pre, t)
            
            vol = fem.assemble_scalar(fem.form(u_crt*ufl.dx))
            ref_area = np.pi*0.2**2
            print(f"Error of area is {(vol - ref_area)/ref_area*100}%")
        

if __name__ == '__main__':
    simulation()

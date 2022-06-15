import numpy as np
import ufl
import dolfinx
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
import glob
import os
import sys
import basix
from pprint import pprint
from src.arguments import args
from src.utils import walltime

comm = MPI.COMM_WORLD

def mpi_print(msg):
    if comm.rank == 0:
        print(f"Rank {comm.rank} print: {msg}")
        sys.stdout.flush()


@walltime
def simulation():
    case_name = 'mechanical'

    if comm.rank == 0:
        files_vtk = glob.glob(f'data/vtk/{case_name}/sols' + f"/*")
        files_xdmf = glob.glob(f'data/xdmf/{case_name}/' + f"/*")
        for f in files_vtk + files_xdmf:
            os.remove(f)

    ambient_T = args.T_ambient
    rho = args.rho
    Cp = args.c_p
    k = args.kappa_T
    h = args.h_conv
    eta = args.power_fraction
    r = args.r_beam
    P = args.power

    x0 = 0.2*args.domain_x
    y0 = 0.5*args.domain_y

    simulation_t = 2400*1e-6
    total_t = 1200*1e-6
    vel = 0.6*args.domain_x/total_t
    dt = 1e-6
    ts = np.arange(0., simulation_t + dt, dt)
    mpi_print(f"total time steps = {len(ts)}")
    ele_size = 0.01

    Nx, Ny, Nz = round(args.domain_x/ele_size), round(args.domain_y/ele_size), round(args.domain_z/ele_size)

    mpi_print(f"Nx = {Nx}, Ny = {Ny}, Nz = {Nz}")

    mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [np.array([0., 0., 0.]), np.array([args.domain_x, args.domain_y, args.domain_z])], 
                                   [Nx, Ny, Nz], cell_type=dolfinx.mesh.CellType.hexahedron)  # cell_type=mesh.CellType.hexahedron/tetrahedron

    mesh_vtk_file = dolfinx.io.VTKFile(MPI.COMM_WORLD, f'data/vtk/{case_name}/mesh/mesh.pvd', 'w')
    mesh_vtk_file.write_mesh(mesh)


    # pprint(dir(mesh.geometry))
    print(f"Total number of local mesh vertices {len(mesh.geometry.x)}" )


    def bottom(x):
        return np.isclose(x[2], 0.)

    def top(x):
        return np.isclose(x[2], args.domain_z)

    fdim = mesh.topology.dim - 1
    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bottom)
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, top)

    mpi_print(f"bottom_facets.shape = {bottom_facets.shape}")

    marked_facets = np.hstack([bottom_facets, top_facets])
    marked_values = np.hstack([np.full(len(bottom_facets), 1, dtype=np.int32), np.full(len(top_facets), 2, dtype=np.int32)])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = dolfinx.mesh.meshtags(mesh, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

    deg_u = 1
    deg_stress = 2
    degree_T = 1

    # "quadrature_degree": 2 means that use 8 integrations ponits for a hexahedron element
    metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
    ds = ufl.Measure('ds', domain=mesh, subdomain_data=facet_tag, metadata=metadata)
    dxm = ufl.Measure('dx', domain=mesh, metadata=metadata)
    normal = ufl.FacetNormal(mesh)
    quadrature_points, wts = basix.make_quadrature(basix.CellType.hexahedron, deg_stress)


    P0_ele = ufl.FiniteElement("DG", mesh.ufl_cell(), 0)
    P0 = fem.FunctionSpace(mesh, P0_ele)
    p_avg = fem.Function(P0, name="Plastic_strain")
    strain_xx = fem.Function(P0, name="strain_xx")
    stress_xx = fem.Function(P0, name="stress_xx")
    phase_avg = fem.Function(P0, name="phase")


    V_ele = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree=degree_T)
    V = fem.FunctionSpace(mesh, V_ele)

    U_ele = ufl.VectorElement("Lagrange", mesh.ufl_cell(), degree=deg_u)
    U = fem.FunctionSpace(mesh, U_ele)

    # W_ele = ufl.TensorElement("DG", mesh.ufl_cell(), 0)
    # W = fem.FunctionSpace(mesh, W_ele)
    # W0_ele = ufl.FiniteElement("DG", mesh.ufl_cell(), 0)
    # W0 = fem.FunctionSpace(mesh, W0_ele)

    # W_ele = ufl.TensorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default', symmetry=True)
    W_ele = ufl.TensorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
    W = fem.FunctionSpace(mesh, W_ele)
    W0_ele = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default') 
    W0 = fem.FunctionSpace(mesh, W0_ele)


    def ini_T(x):
        return np.full(x.shape[1], ambient_T)

    dT = fem.Function(V)

    T_crt = fem.Function(V)
    T_crt.interpolate(ini_T)
    T_pre = fem.Function(V)
    T_pre.interpolate(ini_T)
    T_old = fem.Function(V, name='T')
    T_old.interpolate(ini_T)

    T_trial = ufl.TrialFunction(V) 
    T_test = ufl.TestFunction(V)

    phase = fem.Function(V, name='phase')
    alpha_V = fem.Function(V)
    E = fem.Function(V)

    # alpha_V.x.array[:] = 1e-5
    # E.x.array[:] = args.Young_mod

    nu = 0.3
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2./(1+nu)
    sig0 = 250.
    Et = E/100.  
    H = E*Et/(E-Et)  


    sig = fem.Function(W)
    # Something like "Cumulative plastic strain" may cause an error due to the space - probably a bug of dolfinx
    cumulative_p = fem.Function(W0, name="Cumulative_plastic_strain")
    u = fem.Function(U, name="Total_displacement")
    du = fem.Function(U, name="Iteration_correction")
    Du = fem.Function(U, name="Current_increment")
    v = ufl.TrialFunction(U)
    u_ = ufl.TestFunction(U)

    mpi_print(f"facet_tag.dim = {facet_tag.dim}")
   
    bottom_dofs_u = fem.locate_dofs_topological(U, facet_tag.dim, bottom_facets)
    bcs_u = [fem.dirichletbc(PETSc.ScalarType((0., 0., 0.)), bottom_dofs_u, U)]

    def eps(v):
        e = ufl.sym(ufl.grad(v))
        return e

    def sigma(eps_el):
        return lmbda*ufl.tr(eps_el)*ufl.Identity(3) + 2*mu*eps_el

    deps = eps(Du)

    def thermal_strain():
        # alpha_V = 1e-5
        return alpha_V*dT*ufl.Identity(3)

    ppos = lambda x: (x + abs(x))/2.
    heaviside = lambda x: ufl.conditional(ufl.gt(x, 0.), 1., 0.)

    def proj_sig():
        EPS = 1e-10
        d_eps_T = thermal_strain()
        sig_elas = sig + sigma(deps - d_eps_T)
        s = ufl.dev(sig_elas)
        sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
        f_elas = sig_eq - sig0 - H*cumulative_p
        dp = ppos(f_elas)/(3*mu+H)
        # Prevent divided by zero error
        # The original example (https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html)
        # didn't consider this, and can cause nan error in the solver.
        n_elas = s/(sig_eq + EPS)*heaviside(f_elas)
        beta = 3*mu*dp/(sig_eq + EPS)
        new_sig = sig_elas - beta*s

        return new_sig, n_elas, beta, dp

    def sigma_tang(e):
        return sigma(e) - 3*mu*(3*mu/(3*mu+H)-beta)*ufl.inner(n_elas, e)*n_elas  -2*mu*beta*ufl.dev(e)


    # If theta = 0., we recover implicit Eulear; if theta = 1., we recover explicit Euler; theta = 0.5 seems to be a good choice.
    theta = 0.5
    T_rhs = theta*T_pre + (1 - theta)*T_trial

    bottom_dofs_T = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.indices[facet_tag.values==1])
    bcs_T = [fem.dirichletbc(PETSc.ScalarType(ambient_T), bottom_dofs_T, V)]

    x = ufl.SpatialCoordinate(mesh)
    crt_time = fem.Constant(mesh, PETSc.ScalarType(0.))

    q_laser = 2*P*eta/(np.pi*r**2) * ufl.exp(-2*((x[0] - x0 - vel*crt_time)**2 + (x[1] - y0)**2) / r**2) * ufl.conditional(ufl.gt(crt_time, total_t), 0., 1.)
    # q_laser = 2*P*eta/(np.pi*r**2) * ufl.exp(-2*((x[0] - x0 - vel*crt_time)**2 + (x[1] - y0)**2) / r**2) * heaviside(total_t - crt_time.value)


    q_convection = h * (T_rhs - ambient_T)
    res_T = rho*Cp/dt*(T_trial - T_pre) * T_test * dxm + k * ufl.dot(ufl.grad(T_rhs), ufl.grad(T_test)) * dxm \
                - q_laser * T_test * ds(2) - q_convection * T_test * ds


    new_sig, n_elas, beta, dp = proj_sig()

    # ufl diff might be used to automate the computation of tangent stiffness tensor
    res_u_lhs = ufl.inner(eps(v), sigma_tang(eps(u_)))*dxm
    res_u_rhs = -ufl.inner(new_sig, eps(u_))*dxm  


    problem_T = fem.petsc.LinearProblem(ufl.lhs(res_T), ufl.rhs(res_T), bcs=bcs_T, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    problem_u = fem.petsc.LinearProblem(res_u_lhs, res_u_rhs, bcs=bcs_u, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    def l2_projection(v, V):
        dv = ufl.TrialFunction(V)
        v_ = ufl.TestFunction(V)
        a_proj = ufl.inner(dv, v_)*dxm
        b_proj = ufl.inner(v, v_)*dxm
        problem = fem.petsc.LinearProblem(a_proj, b_proj, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u = problem.solve()
        return u


    def local_projection(v, V):
        '''
        See https://github.com/FEniCS/dolfinx/issues/2243
        '''
        u = fem.Function(V)
        e_expr = fem.Expression(v, quadrature_points)
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)
        e_eval = e_expr.eval(cells)

        with u.vector.localForm() as u_local:
            u_local.setBlockSize(u.function_space.dofmap.bs)
            u_local.setValuesBlocked(V.dofmap.list.array, e_eval, addv=PETSc.InsertMode.INSERT)

        return u


    def update_modului():
        # 0: powder, 1: liquid, 2: solid 
        T_array = T_crt.x.array

        powder_to_liquid = (phase.x.array == 0) & (T_array > args.T_melt)
        liquid_to_solid = (phase.x.array == 1) & (T_array < args.T_melt)

        phase.x.array[powder_to_liquid] = 1
        phase.x.array[liquid_to_solid] = 2

        # print(f"number of powder = {np.sum(phase.x.array == 0)}, liquid = {np.sum(phase.x.array == 1)}, solid = {np.sum(phase.x.array == 2)}")

        E.x.array[(phase.x.array == 0) | (phase.x.array == 1)]  = 1e-2*args.Young_mod 
        E.x.array[phase.x.array == 2] = args.Young_mod 

        alpha_V.x.array[(phase.x.array == 0) | (phase.x.array == 1)] = 0. 
 
        alpha_V.x.array[phase.x.array == 2] = args.alpha_V
  
    def write_sol(file, step):
        file.write_function(T_old, step)
        file.write_function(u, step)
        file.write_function(p_avg, step)
        file.write_function(strain_xx, step) 
        file.write_function(stress_xx, step)     
        file.write_function(phase_avg, step)

    vtk_file = dolfinx.io.VTKFile(mesh.comm, f'data/vtk/{case_name}/sols/u.pvd', 'w')
    xdmf_file = dolfinx.io.XDMFFile(mesh.comm, f'data/xdmf/{case_name}/u.xdmf', 'w')

    xdmf_file.write_mesh(mesh)

    write_sol(vtk_file, 0)
    write_sol(xdmf_file, 0)

    plastic_inverval = 20

    for i in range(len(ts) - 1):
    # for i in range(20):

        mpi_print(f"step {i + 1}/{len(ts) - 1}, time = {ts[i + 1]}")
        crt_time.value = theta*ts[i] + (1 - theta)*ts[i + 1]

        update_modului()

        T_crt = problem_T.solve()
 
        T_pre.x.array[:] = T_crt.x.array

        # print(f"min T = {np.min(np.array(T_pre.x.array))}")
        # print(f"max T = {np.max(np.array(T_pre.x.array))}\n")

        if (i + 1) % plastic_inverval == 0:

            T_crt_array = np.array(T_crt.x.array)
            T_crt_array = np.where(T_crt_array < args.T_ambient, args.T_ambient, T_crt_array)
            T_crt_array = np.where(T_crt_array > args.T_melt, args.T_melt, T_crt_array)
            T_old_array = np.array(T_old.x.array)
            T_old_array = np.where(T_old_array < args.T_ambient, args.T_ambient, T_old_array)
            T_old_array = np.where(T_old_array > args.T_melt, args.T_melt, T_old_array)
            dT.x.array[:] = T_crt_array - T_old_array

            Du.x.array[:] = 0.

            niter = 0
            nRes = 1.
            tol = 1e-8

            while nRes > tol or niter < 2:
                mpi_print(f"At iteration step {niter + 1}")
                du = problem_u.solve()
                Du.x.array[:] = Du.x.array + du.x.array
                mpi_print(f"du norm = {np.linalg.norm(du.x.array)}")

                # nRes1 = np.sqrt(mesh.comm.allreduce(np.sum(problem_u.b.array**2), op=MPI.SUM))
                nRes = problem_u.b.norm(1)
                mpi_print(f"b norm: {nRes}\n")
                niter += 1

            u.x.array[:] = u.x.array + Du.x.array

            sig.x.array[:] = local_projection(new_sig, W).x.array
            mpi_print(f"sig dof = {sig.x.array.shape}")
            mpi_print(f"sig norm = {np.linalg.norm(sig.x.array)}")

            cumulative_p.x.array[:] = cumulative_p.x.array + local_projection(dp, W0).x.array

            # Remark: Can we do interpolation here?
            # p_avg.interpolate(fem.Expression(cumulative_p, P0.element.interpolation_points))
            p_avg.x.array[:] = l2_projection(cumulative_p, P0).x.array
            strain_xx.x.array[:] = l2_projection(ufl.grad(u)[0, 0], P0).x.array
            stress_xx.x.array[:] = l2_projection(sig[0, 0], P0).x.array
            phase_avg.x.array[:] = l2_projection(phase, P0).x.array

            T_old.x.array[:] = T_crt.x.array

            write_sol(vtk_file, i + 1)
            write_sol(xdmf_file, i + 1)


if __name__ == '__main__':
    simulation()

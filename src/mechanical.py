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

    mesh_vtk_file = io.VTKFile(MPI.COMM_WORLD, f'data/vtk/{case_name}/mesh/mesh.pvd', 'w')
    mesh_vtk_file.write_mesh(domain)

    E = 70e3
    nu = 0.3
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2./(1+nu)
    sig0 = 250.
    Et = E/100.  
    H = E*Et/(E-Et)  

    def bottom(x):
        return np.isclose(x[2], 0.)

    def top(x):
        return np.isclose(x[2], args.domain_z)

    fdim = domain.topology.dim - 1
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
    top_facets = mesh.locate_entities_boundary(domain, fdim, top)

    print(f"bottom_facets.shape = {bottom_facets.shape}")

    marked_facets = np.hstack([bottom_facets, top_facets])
    marked_values = np.hstack([np.full(len(bottom_facets), 1, dtype=np.int32), np.full(len(top_facets), 2, dtype=np.int32)])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

    # TODO: how to determine quadrature_degree?
    metadata = {"quadrature_degree": 2, "quadrature_scheme": "default"}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
    dxm = ufl.Measure('dx', domain=domain, metadata=metadata)
    normal = ufl.FacetNormal(domain)

    U_ele = ufl.VectorElement("Lagrange", domain.ufl_cell(), degree=1)
    U = fem.FunctionSpace(domain, U_ele)


    W_ele = ufl.TensorElement("DG", domain.ufl_cell(), 0)
    W = fem.FunctionSpace(domain, W_ele)
    W0_ele = ufl.FiniteElement("DG", domain.ufl_cell(), 0)
    W0 = fem.FunctionSpace(domain, W0_ele)


    # W0_ele = ufl.FiniteElement("Quadrature", domain.ufl_cell(), degree=2, quad_scheme='default')
    # W0 = fem.FunctionSpace(domain, W0_ele)
    # W_ele = ufl.TensorElement("Quadrature", domain.ufl_cell(), degree=2, quad_scheme='default')
    # W = fem.FunctionSpace(domain, W_ele)

    sig = fem.Function(W)
    sig_old = fem.Function(W)
    n_elas = fem.Function(W)
    beta = fem.Function(W0)
    # Something like "Cumulative plastic strain" may cause an error due to the space - probably a bug of dolfinx
    cumulative_p = fem.Function(W0, name="Cumulative_plastic_strain")
    u = fem.Function(U, name="Total_displacement")
    du = fem.Function(U, name="Iteration_correction")
    Du = fem.Function(U, name="Current_increment")
    v = ufl.TrialFunction(U)
    u_ = ufl.TestFunction(U)


    print(f"facet_tag.dim = {facet_tag.dim}")

   
    bottom_dofs_u = fem.locate_dofs_topological(U, facet_tag.dim, bottom_facets)
    bcs_u = [fem.dirichletbc(PETSc.ScalarType((0., 0., 0.)), bottom_dofs_u, U)]

    def eps(v):
        e = ufl.sym(ufl.grad(v))
        return e

    def sigma(eps_el):
        return lmbda*ufl.tr(eps_el)*ufl.Identity(3) + 2*mu*eps_el


    ppos = lambda x: (x + abs(x))/2.

    def thermal_strain(dT):
        alpha_V = 1e-5
        return alpha_V*dT*ufl.Identity(3)

    def proj_sig(deps, dT, sig_old, p_old):
        sig_n = sig_old

        d_eps_T = thermal_strain(dT)
        sig_elas = sig_n + sigma(deps - d_eps_T)

        s = ufl.dev(sig_elas)
        sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
        f_elas = sig_eq - sig0 - H*p_old
        dp = ppos(f_elas)/(3*mu+H)
        n_elas = s/sig_eq*ppos(f_elas)/f_elas
        beta = 3*mu*dp/sig_eq
        new_sig = sig_elas - beta*s

        return new_sig, n_elas, beta, dp

 
    def sigma_tang(e):
        return sigma(e) - 3*mu*(3*mu/(3*mu+H)-beta)*ufl.inner(n_elas, e)*n_elas-2*mu*beta*ufl.dev(e)

    # TODO
    a_Newton = ufl.inner(eps(v), sigma_tang(eps(u_)))*dxm
    res = -ufl.inner(sig, eps(u_))*dxm + 1e-10*ufl.dot(normal, u_)*ds(2)

    F_u = a_Newton - res

    P0_ele = ufl.FiniteElement("DG", domain.ufl_cell(), 0)
    P0 = fem.FunctionSpace(domain, P0_ele)
    p_avg = fem.Function(P0, name="Plastic_strain")


    V_ele = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree=1)
    V = fem.FunctionSpace(domain, V_ele)

    def ini_T(x):
        return np.full(x.shape[1], ambient_T)

    T_crt = fem.Function(V)
    T_crt.interpolate(ini_T)
    T_pre = fem.Function(V)
    T_pre.interpolate(ini_T)
    T_old = fem.Function(V, name='T')
    T_old.interpolate(ini_T)

    T_trial = ufl.TrialFunction(V) 
    T_test = ufl.TestFunction(V)

    # If theta = 0., we recover implicit Eulear; if theta = 1., we recover explicit Euler; theta = 0.5 seems to be a good choice.
    theta = 0.5
    T_rhs = theta*T_pre + (1 - theta)*T_trial

    bottom_dofs_T = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.indices[facet_tag.values==1])
    bcs_T = [fem.dirichletbc(PETSc.ScalarType(ambient_T), bottom_dofs_T, V)]

    x = ufl.SpatialCoordinate(domain)
    crt_time = fem.Constant(domain, PETSc.ScalarType(0.))
    q_laser = 2*P*eta/(np.pi*r**2) * ufl.exp(-2*((x[0] - x0 - vel*crt_time)**2 + (x[1] - y0)**2) / r**2)
    q_convection = h * (T_rhs - ambient_T)
    res_T = rho*Cp/dt*(T_trial - T_pre) * T_test * ufl.dx + k * ufl.dot(ufl.grad(T_rhs), ufl.grad(T_test)) * ufl.dx \
                - q_laser * T_test * ds(2) - q_convection * T_test * ufl.ds



    files_vtk = glob.glob(f'data/vtk/{case_name}/sols' + f"/*")
    files_xdmf = glob.glob(f'data/xdmf/{case_name}/' + f"/*")
    for f in files_vtk + files_xdmf:
        os.remove(f)


    T_vtk_file = io.VTKFile(MPI.COMM_WORLD, f'data/vtk/{case_name}/sols/T.pvd', 'w')
    u_vtk_file = io.VTKFile(MPI.COMM_WORLD, f'data/vtk/{case_name}/sols/u.pvd', 'w')
    p_vtk_file = io.VTKFile(MPI.COMM_WORLD, f'data/vtk/{case_name}/sols/p.pvd', 'w')
    T_vtk_file.write_function(T_old, crt_time.value)
    u_vtk_file.write_function(u, crt_time.value)
    p_vtk_file.write_function(p_avg, crt_time.value)


    xdmf_file = io.XDMFFile(MPI.COMM_WORLD, f'data/xdmf/{case_name}/u.xdmf', 'w')
    xdmf_file.write_mesh(domain)
    xdmf_file.write_function(T_old, 0)
    xdmf_file.write_function(u, 0)
    xdmf_file.write_function(p_avg, 0)

    plastic_inverval = 10

    problem_T = fem.petsc.LinearProblem(ufl.lhs(res_T), ufl.rhs(res_T), bcs=bcs_T, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    problem_u = fem.petsc.LinearProblem(a_Newton, res, bcs=bcs_u, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})


    def l2_projection(v, V):
        dv = ufl.TrialFunction(V)
        v_ = ufl.TestFunction(V)
        a_proj = ufl.inner(dv, v_)*dxm
        b_proj = ufl.inner(v, v_)*dxm
        problem = fem.petsc.LinearProblem(a_proj, b_proj, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u = problem.solve()
        return u
 

    for i in range(len(ts) - 1):
    # for i in range(100):

        print(f"step {i + 1}/{len(ts) - 1}, time = {ts[i + 1]}")
        crt_time.value = theta*ts[i] + (1 - theta)*ts[i + 1]

        T_crt = problem_T.solve()

        # TODO: check if no [:] works
        T_pre.x.array[:] = T_crt.x.array

        print(f"min T = {np.min(np.array(T_pre.x.array))}")
        print(f"max T = {np.max(np.array(T_pre.x.array))}")

        if (i + 1) % plastic_inverval == 0:

            T_crt_array = np.array(T_crt.x.array)
            T_crt_array = np.where(T_crt_array < args.T_ambient, args.T_ambient, T_crt_array)
            T_crt_array = np.where(T_crt_array > args.T_melt, args.T_melt, T_crt_array)
            T_old_array = np.array(T_old.x.array)
            T_old_array = np.where(T_old_array < args.T_ambient, args.T_ambient, T_old_array)
            T_old_array = np.where(T_old_array > args.T_melt, args.T_melt, T_old_array)
            dT = fem.Function(V)
            dT.x.array[:] = T_crt_array - T_old_array

            Du = fem.Function(U)
      
            niter = 0
            nRes = 1.
            tol = 1e-8

            while nRes > tol or True:
                print("\n")

                du = problem_u.solve()
 
                Du.x.array[:] = Du.x.array + du.x.array
                deps = eps(Du)
                print(f"du norm = {np.linalg.norm(du.x.array)}")

                sig_, n_elas_, beta_, dp_ = proj_sig(deps, dT, sig_old, cumulative_p)
          
                sig.x.array[:] = l2_projection(sig_, W).x.array
                n_elas.x.array[:] = l2_projection(n_elas_, W).x.array
                beta.x.array[:] = l2_projection(beta_, W0).x.array

                print(f"sig dof = {sig.x.array.shape}")
                print(f"sig norm = {np.linalg.norm(sig.x.array)}")
                print(f"n_elas norm = {np.linalg.norm(n_elas.x.array)}")
                print(f"beta norm = {np.linalg.norm(beta.x.array)}")
              
                # nRes = np.sqrt(domain.comm.allreduce(np.sum(b.array**2), op=MPI.SUM))
                nRes = problem_u.b.norm(1)
                print(f"b norm: {nRes}")
                niter += 1

            u.x.array[:] = u.x.array + Du.x.array
            sig_old.x.array[:] = sig.x.array

            # cumulative_p.x.array[:] = cumulative_p.x.array + l2_projection(dp_, W0).x.array

            p_avg.x.array[:] = l2_projection(cumulative_p, P0).x.array

            T_old.x.array[:] = T_crt.x.array


            T_vtk_file.write_function(T_old, i)
            u_vtk_file.write_function(u, i)
            p_vtk_file.write_function(p_avg, i)
            xdmf_file.write_function(T_old, i)
            xdmf_file.write_function(u, i)
            xdmf_file.write_function(p_avg, i)



if __name__ == '__main__':
    simulation()

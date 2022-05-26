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


    # metadata = {"quadrature_degree": 4}
    metadata = {}
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
    # dx = ufl.Measure('dx', domain=domain, metadata=metadata)
    normal = ufl.FacetNormal(domain)

    deg_u = 2
    deg_stress = 2
    # TODO: can we do a tensor element?
    U_ele = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
    U = fem.FunctionSpace(domain, U_ele)

    W_ele = ufl.VectorElement("Quadrature", domain.ufl_cell(), degree=deg_stress, dim=9, quad_scheme='default')
    W = fem.FunctionSpace(domain, W_ele)
    W0_ele = ufl.FiniteElement("Quadrature", domain.ufl_cell(), degree=deg_stress, quad_scheme='default')
    W0 = fem.FunctionSpace(domain, W0_ele)


    sig = fem.Function(W)
    sig_old = fem.Function(W)
    n_elas = fem.Function(W)
    beta = fem.Function(W0)
    cumulative_p = fem.Function(W0, name="Cumulative plastic strain")
    u = fem.Function(U, name="Total displacement")
    du = fem.Function(U, name="Iteration correction")
    Du = fem.Function(U, name="Current increment")
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

    def as_3D_tensor(X):
        return ufl.as_tensor([[X[0], X[1], X[2]],
                              [X[3], X[4], X[5]],
                              [X[6], X[7], X[8]]])

    def as_long_vector(X):
        return ufl.as_vector([X[0, 0], X[0, 1], X[0, 2], X[1, 0], X[1, 1], X[1, 2], X[2, 0], X[2, 1], X[2, 2]])


    ppos = lambda x: (x + abs(x))/2.

    def thermal_strain(dT):
        alpha_V = 1e-5
        return alpha_V*dT*ufl.Identity(3)

    def proj_sig(deps, dT, sig_old, p_old):
        sig_n = as_3D_tensor(sig_old)

        # sig_elas = sig_n + sigma(deps)

        d_eps_T = thermal_strain(dT)
        sig_elas = sig_n + sigma(deps - d_eps_T)

        s = ufl.dev(sig_elas)
        sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
        f_elas = sig_eq - sig0 - H*p_old
        dp = ppos(f_elas)/(3*mu+H)
        n_elas = s/sig_eq*ppos(f_elas)/f_elas
        beta = 3*mu*dp/sig_eq
        new_sig = sig_elas - beta*s

        return as_long_vector(new_sig), as_long_vector(n_elas), beta, dp

 
    def sigma_tang(e):
        N_elas = as_3D_tensor(n_elas)
        return sigma(e) - 3*mu*(3*mu/(3*mu+H)-beta)*ufl.inner(N_elas, e)*N_elas-2*mu*beta*ufl.dev(e)


    metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
    dxm = ufl.dx(metadata=metadata)

    # TODO
    a_Newton = ufl.inner(eps(v), sigma_tang(eps(u_)))*dxm
    res = -ufl.inner(eps(u_), as_3D_tensor(sig))*dxm + 1e-10*ufl.dot(normal, u_)*ds(2)

    P0_ele = ufl.FiniteElement("DG", domain.ufl_cell(), 0)
    P0 = fem.FunctionSpace(domain, P0_ele)
    p_avg = fem.Function(P0, name="Plastic strain")


    V_ele = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
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
    # T_rhs = theta*T_pre + (1 - theta)*T_crt

    bottom_dofs_T = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.indices[facet_tag.values==1])
    bcs_T = [fem.dirichletbc(PETSc.ScalarType(ambient_T), bottom_dofs_T, V)]


    x = ufl.SpatialCoordinate(domain)

    crt_time = fem.Constant(domain, PETSc.ScalarType(0.))
    q_laser = 2*P*eta/(np.pi*r**2) * ufl.exp(-2*((x[0] - x0 - vel*crt_time)**2 + (x[1] - y0)**2) / r**2)
    # TODO: q_convection = h * (T_rhs - ambient_T)
    q_convection = h * (T_pre - ambient_T)
 
    # res_T = rho*Cp/dt*(T_crt - T_pre) * v * ufl.dx + k * ufl.dot(ufl.grad(T_rhs), ufl.grad(v)) * ufl.dx \
    #             - q_laser * v * ds(2) - q_convection * v * ufl.ds

    res_T_left = rho*Cp/dt*T_trial * T_test * ufl.dx + k * (1 - theta) * ufl.dot(ufl.grad(T_trial), ufl.grad(T_test)) * ufl.dx
    res_T_right = rho*Cp/dt*T_pre * T_test * ufl.dx - k * theta * ufl.dot(ufl.grad(T_pre), ufl.grad(T_test)) * ufl.dx \
                  + q_laser * T_test * ds(2) + q_convection * T_test * ufl.ds


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

    problem_T = fem.petsc.LinearProblem(res_T_left, res_T_right, bcs=bcs_T, petsc_options={"ksp_type": "preonly"})

    problem_u = fem.petsc.LinearProblem(a_Newton, res, bcs=bcs_u, petsc_options={"ksp_type": "preonly"})


 

    def local_project(v, V):

        v = 3.*v
        # a = fem.Function(W0)  
        # b = triple(a)
        # v = H*b

        dv = ufl.TrialFunction(V)
        v_ = ufl.TestFunction(V)

        # a_proj = ufl.inner(dv, v_)*dxm
        # b_proj = ufl.inner(v, v_)*dxm
        a_proj = dv*v_*dxm
        b_proj = 3.*v*v_*dxm

        problem = fem.petsc.LinearProblem(a_proj, b_proj, bcs=[], petsc_options={"ksp_type": "preonly"})
        u = problem.solve()
        return u
 



    def proj_debug(deps, dT, sig_old, p_old):
        sig_n = as_3D_tensor(sig_old)

        # sig_elas = sig_n + sigma(deps)

        d_eps_T = thermal_strain(dT)
        sig_elas = sig_n + sigma(deps - d_eps_T)

        s = ufl.dev(sig_elas)
        sig_eq = ufl.sqrt(3/2.*ufl.inner(s, s))
        f_elas = sig_eq - sig0 - H*p_old

        f_elas = H*p_old


        dp = ppos(f_elas)/(3*mu+H)
        n_elas = s/sig_eq*ppos(f_elas)/f_elas
        beta = 3*mu*dp/sig_eq
        new_sig = sig_elas - beta*s

        return f_elas



    for i in range(len(ts) - 1):
    # for i in range(100):

        print(f"step {i + 1}, time = {ts[i + 1]}")
        crt_time.value = theta*ts[i] + (1 - theta)*ts[i + 1]

        T_crt = problem_T.solve()
        # TODO: check if no [:] works
        T_pre.x.array[:] = T_crt.x.array
      
        # print(f"min T = {np.min(np.array(T_pre.vector()))}")
        # print(f"max T = {np.max(np.array(T_pre.vector()))}")

        if (i + 1) % plastic_inverval == 0:

            T_crt_array = np.array(T_crt.x.array)
            T_crt_array = np.where(T_crt_array < args.T_ambient, args.T_ambient, T_crt_array)
            T_crt_array = np.where(T_crt_array > args.T_melt, args.T_melt, T_crt_array)
            T_old_array = np.array(T_old.x.array)
            T_old_array = np.where(T_old_array < args.T_ambient, args.T_ambient, T_old_array)
            T_old_array = np.where(T_old_array > args.T_melt, args.T_melt, T_old_array)
            dT = fem.Function(V)
            dT.x.array[:] = T_crt_array - T_old_array

            # exit()
        
            Du = fem.Function(U)
      
            niter = 0
            nRes = 1.
            tol = 1e-8

            while nRes > tol:

                du = problem_u.solve()


                Du.x.array[:] = Du.x.array + du.x.array
                deps = eps(Du)

                sig_, n_elas_, beta_, dp_ = proj_sig(deps, dT, sig_old, cumulative_p)
                var = proj_debug(deps, dT, sig_old, cumulative_p)


                # cumulative_p = fem.Function(W0)

                local_project(H*cumulative_p, W0)

                exit()
        
                beta = local_project(beta_, W0)


                sig = local_project(sig_, W)
                n_elas = local_project(n_elas_, W)

                # TODO: beta = local_project(beta_, W0) is buggy


                nRes = np.sqrt(domain.comm.allreduce(np.sum(u_D.x.array**2), op=MPI.SUM))
                print(f"Residual: {nRes}")
                niter += 1



            u.x.array[:] = u.x.array + Du.x.array
            sig_old.x.array[:] = sig.x.array

            dp = local_project(dp_, W0)
            p.x.array[:] = p.x.array + dp.x.array

            p_avg.x.array[:] = local_project(p, P0).x.array
   
        

            T_old.x.array[:] = T_crt.x.array



            T_vtk_file.write_function(T_old, i)
            u_vtk_file.write_function(u, i)
            p_vtk_file.write_function(p_avg, i)
            xdmf_file.write_function(T_old, i)
            xdmf_file.write_function(u, i)
            xdmf_file.write_function(p_avg, i)


  



if __name__ == '__main__':
    simulation()

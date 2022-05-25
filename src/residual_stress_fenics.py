import numpy as np
from src.arguments import args
from src.utils import walltime
import fenics as fe
import os
import glob

fe.parameters["form_compiler"]["representation"] = 'quadrature'
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)


@walltime
def simulation():
    ambient_T = args.T_ambient
    rho = args.rho
    Cp = args.c_p
    k = args.kappa_T
    h = args.h_conv
    eta = args.power_fraction
    r = args.r_beam
    P = args.power
    EPS = 1e-8


    E = fe.Constant(70e3)
    nu = fe.Constant(0.3)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2./(1+nu)
    sig0 = fe.Constant(250.)  
    Et = E/100.  
    H = E*Et/(E-Et)  


    x0 = 0.2*args.domain_x
    y0 = 0.5*args.domain_y

    total_t = 1200*1e-6
    vel = 0.6*args.domain_x/total_t
    dt = 1e-6
    ts = np.arange(0., total_t + dt, dt)
    print(f"total time steps = {len(ts)}")

    ele_size = 0.01

    mesh = fe.BoxMesh(fe.Point(0., 0., 0.), fe.Point(args.domain_x, args.domain_y, args.domain_z), 
                      round(args.domain_x/ele_size), round(args.domain_y/ele_size), round(args.domain_z/ele_size))
    mesh_file = fe.File(f'data/vtk/pbf/mesh/mesh.pvd')
    mesh_file << mesh

    # Define bottom surface 
    class Bottom(fe.SubDomain):
        def inside(self, x, on_boundary):
            # The condition for a point x to be on bottom side is that x[2] < EPS
            return on_boundary and x[2] < EPS

    # Define top surface
    class Top(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[2] > args.domain_z - EPS

    # Define the other four surfaces
    class SurroundingSurfaces(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (x[0] < EPS or x[0] > args.domain_x - EPS or x[1] < EPS or x[1] > args.domain_y - EPS)

    bottom = Bottom()
    top = Top()
    surrounding_surfaces = SurroundingSurfaces()
    boundaries = fe.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    bottom.mark(boundaries, 1)
    top.mark(boundaries, 2)
    surrounding_surfaces.mark(boundaries, 3)
    ds = fe.Measure('ds')(subdomain_data=boundaries)
    normal = fe.FacetNormal(mesh)


    deg_u = 2
    deg_stress = 2
    # TODO: can we do a tensor element?
    U = fe.VectorFunctionSpace(mesh, "CG", deg_u)
    We = fe.VectorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, dim=9, quad_scheme='default')
    W = fe.FunctionSpace(mesh, We)
    # We = fe.TensorElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
    # W = fe.FunctionSpace(mesh, We)
    W0e = fe.FiniteElement("Quadrature", mesh.ufl_cell(), degree=deg_stress, quad_scheme='default')
    W0 = fe.FunctionSpace(mesh, W0e)

    sig = fe.Function(W)
    sig_old = fe.Function(W)
    n_elas = fe.Function(W)
    beta = fe.Function(W0)
    p = fe.Function(W0, name="Cumulative plastic strain")
    u = fe.Function(U, name="Total displacement")
    du = fe.Function(U, name="Iteration correction")
    Du = fe.Function(U, name="Current increment")
    v = fe.TrialFunction(U)
    u_ = fe.TestFunction(U)

    u_bcs = [fe.DirichletBC(U, fe.Constant((0., 0., 0.)), bottom)]

    def eps(v):
        e = fe.sym(fe.grad(v))
        return e

    def sigma(eps_el):
        return lmbda*fe.tr(eps_el)*fe.Identity(3) + 2*mu*eps_el

    def as_3D_tensor(X):
        return fe.as_tensor([[X[0], X[1], X[2]],
                             [X[3], X[4], X[5]],
                             [X[6], X[7], X[8]]])

    def as_long_vector(X):
        return fe.as_vector([X[0, 0], X[0, 1], X[0, 2], X[1, 0], X[1, 1], X[1, 2], X[2, 0], X[2, 1], X[2, 2]])


    ppos = lambda x: (x+abs(x))/2.

    def thermal_strain(dT):
        alpha_V = 1e-5
        return alpha_V*dT*fe.Identity(3)

    def proj_sig(deps, dT, sig_old, p_old):
        sig_n = as_3D_tensor(sig_old)

        # sig_elas = sig_n + sigma(deps)

        d_eps_T = thermal_strain(dT)
        sig_elas = sig_n + sigma(deps - d_eps_T)

        s = fe.dev(sig_elas)
        sig_eq = fe.sqrt(3/2.*fe.inner(s, s))
        f_elas = sig_eq - sig0 - H*p_old
        dp = ppos(f_elas)/(3*mu+H)
        n_elas = s/sig_eq*ppos(f_elas)/f_elas
        beta = 3*mu*dp/sig_eq
        new_sig = sig_elas-beta*s

        return as_long_vector(new_sig), as_long_vector(n_elas), beta, dp

 
    def sigma_tang(e):
        N_elas = as_3D_tensor(n_elas)
        return sigma(e) - 3*mu*(3*mu/(3*mu+H)-beta)*fe.inner(N_elas, e)*N_elas-2*mu*beta*fe.dev(e)


    metadata = {"quadrature_degree": deg_stress, "quadrature_scheme": "default"}
    dxm = fe.dx(metadata=metadata)

    # TODO
    a_Newton = fe.inner(eps(v), sigma_tang(eps(u_)))*dxm
    res = -fe.inner(eps(u_), as_3D_tensor(sig))*dxm + 1e-10*fe.dot(normal, u_)*ds(2)

    def local_project(v, V, u=None):
        dv = fe.TrialFunction(V)
        v_ = fe.TestFunction(V)
        a_proj = fe.inner(dv, v_)*dxm
        b_proj = fe.inner(v, v_)*dxm
        solver = fe.LocalSolver(a_proj, b_proj)
        solver.factorize()
        if u is None:
            u = fe.Function(V)
            solver.solve_local_rhs(u)
            return u
        else:
            solver.solve_local_rhs(u)


    P0 = fe.FunctionSpace(mesh, "DG", 0)
    p_avg = fe.Function(P0, name="Plastic strain")


    V = fe.FunctionSpace(mesh, 'CG', 1)
    T_crt = fe.interpolate(fe.Constant(ambient_T), V)
    T_pre = fe.interpolate(fe.Constant(ambient_T), V)
    T_old = fe.interpolate(fe.Constant(ambient_T), V)
    T_old.rename('T', 'T')
    v = fe.TestFunction(V)

    # If theta = 0., we recover implicit Eulear; if theta = 1., we recover explicit Euler; theta = 0.5 seems to be a good choice.
    theta = 1.
    T_rhs = theta*T_pre + (1 - theta)*T_crt
    T_bcs = [fe.DirichletBC(V, fe.Constant(ambient_T), bottom)]

    class LaserExpression(fe.UserExpression):
        def __init__(self, t):
            super(LaserExpression, self).__init__()
            self.t = t

        def eval(self, values, x):
            t = self.t
            values[0] = 2*P*eta/(np.pi*r**2) * np.exp(-2*((x[0] - x0 - vel*t)**2 + (x[1] - y0)**2) / r**2)
    
        def value_shape(self):
            return ()

    q_laser = LaserExpression(None)
    q_convection = h * (T_rhs - ambient_T)

    q_top = q_convection + q_laser 
    q_surr = q_convection

    res_T = rho*Cp/dt*(T_crt - T_pre) * v * fe.dx + k * fe.dot(fe.grad(T_rhs), fe.grad(v)) * fe.dx \
                - q_top * v * ds(2) - q_surr * v * ds(3)


    files_vtk = glob.glob('data/vtk/pbf/sols' + f"/*")
    files_xdmf = glob.glob('data/xdmf/pbf/' + f"/*")
    for f in files_vtk + files_xdmf:
        os.remove(f)

    T_vtk_file = fe.File(f'data/vtk/pbf/sols/T.pvd')
    T_vtk_file << T_old
    u_vtk_file = fe.File(f'data/vtk/pbf/sols/u.pvd')
    u_vtk_file << u
    p_vtk_file = fe.File(f'data/vtk/pbf/sols/p.pvd')
    p_vtk_file << p_avg

    file_results = fe.XDMFFile('data/xdmf/pbf/u.xdmf')
    file_results.parameters["functions_share_mesh"] = True
    file_results.write(T_old, 0)
    file_results.write(u, 0)
    file_results.write(p_avg, 0)

    plastic_inverval = 10

    for i in range(len(ts) - 1):
    # for i in range(100):

        print(f"step {i + 1}, time = {ts[i + 1]}")
        q_laser.t = theta*ts[i] + (1 - theta)*ts[i + 1]
        solver_parameters = {'newton_solver': {'maximum_iterations': 20, 'linear_solver': 'mumps'}}
        fe.solve(res_T == 0, T_crt, T_bcs, solver_parameters=solver_parameters)
        T_pre.assign(T_crt)
        print(f"min T = {np.min(np.array(T_pre.vector()))}")
        print(f"max T = {np.max(np.array(T_pre.vector()))}")

        if (i + 1) % plastic_inverval == 0:

            T_crt_array = np.array(T_crt.vector())
            T_crt_array = np.where(T_crt_array < args.T_ambient, args.T_ambient, T_crt_array)
            T_crt_array = np.where(T_crt_array > args.T_melt, args.T_melt, T_crt_array)
            T_old_array = np.array(T_old.vector())
            T_old_array = np.where(T_old_array < args.T_ambient, args.T_ambient, T_old_array)
            T_old_array = np.where(T_old_array > args.T_melt, args.T_melt, T_old_array)
            dT = fe.Function(V)
            dT.vector()[:] = T_crt_array - T_old_array

            Nitermax, tol = 200, 1e-8  # parameters of the Newton-Raphson procedure
            Nincr = 20
            A, Res = fe.assemble_system(a_Newton, res, u_bcs)
            nRes0 = Res.norm("l2")
            nRes = nRes0
            Du.interpolate(fe.Constant((0., 0., 0.)))

            niter = 0
            nRes = 1.
            tol = 1e-8
            # while nRes/nRes0 > tol and niter < Nitermax:
            # while niter < Nitermax:
            while nRes > tol:
                fe.solve(A, du.vector(), Res, "mumps")
                Du.assign(Du+du)
                deps = eps(Du)
     
                sig_, n_elas_, beta_, dp_ = proj_sig(deps, dT, sig_old, p)
                local_project(sig_, W, sig)
                local_project(n_elas_, W, n_elas)
                local_project(beta_, W0, beta)
                A, Res = fe.assemble_system(a_Newton, res, u_bcs)
                nRes = Res.norm("l2")
                print(f"Residual: {nRes}")
                niter += 1

            u.assign(u + Du)
            sig_old.assign(sig)
            p.assign(p + local_project(dp_, W0))
            p_avg.assign(fe.project(p, P0))

            T_old.assign(T_crt)


            T_vtk_file << T_old
            u_vtk_file << u
            p_vtk_file << p_avg

            file_results.write(T_old, i)
            file_results.write(u, i)
            file_results.write(p_avg, i)
            

if __name__ == '__main__':
    simulation()

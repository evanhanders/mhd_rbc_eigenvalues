import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import time
import dedalus.public as de
import numpy as np
import os

comm = MPI.COMM_WORLD

def mhd_rbc_evp(Q, Ra, k, Nz=24, noHorizB_BCs=True):
    z = de.Chebyshev('z',Nz, interval=(0, 1))
    d = de.Domain([z],comm=MPI.COMM_SELF)

    variables=['T', 'T_z', 'p','u','w','Oy', 'Bx','By', 'Bz', 'Jz_z']#, 'Bx_z', 'By_z']
#    variables=['T', 'T_z', 'p','u','w','Oy','Ox', 'Bx','By', 'Bz', 'Jx', 'Jx_z']
    problem = de.EVP(d,variables, eigenvalue='omega')

    #Parameters
    problem.parameters['kx']   = k
    problem.parameters['ky']   = 0
    problem.parameters['Ra']   = Ra #Rayleigh number
    problem.parameters['Pr']   = 1   #Prandtl number
    problem.parameters['Pm']   = 1   #Magnetic Prandtl number
    problem.parameters['Q']    = Q 
    problem.parameters['dzT0'] = -1

    #Substitutions
    problem.substitutions['v']     = "0"
    problem.substitutions['Ox']     = "0"
    problem.substitutions["dy(A)"] = "0"

    problem.substitutions["dt(A)"] = "omega*A"
    problem.substitutions["dx(A)"] = "1j*kx*A"
    problem.substitutions['UdotGrad(A,A_z)'] = '(u*dx(A)  + v*dy(A)  + w*(A_z))'
    problem.substitutions['BdotGrad(A,A_z)'] = '(Bx*dx(A) + By*dy(A) + Bz*(A_z))'
    problem.substitutions['Lap(A,A_z)'] = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
    problem.substitutions["Jx"] = "dy(Bz)-dz(By)"
    problem.substitutions["Jy"] = "dz(Bx)-dx(Bz)"
    problem.substitutions["Jz"] = "dx(By)-dy(Bx)"
    problem.substitutions["Oz"] = "dx(v)-dy(u)"
    problem.substitutions["Kz"] = "dx(Oy)-dy(Ox)"
    problem.substitutions["Ky"] = "dz(Ox)-dx(Oz)"
    problem.substitutions["Kx"] = "dy(Oz)-dz(Oy)"


    #Dimensionless parameter substitutions
    problem.substitutions["inv_Re_ff"]    = "(Pr/Ra)**(1./2.)"
    problem.substitutions["inv_Rem_ff"]   = "(inv_Re_ff / Pm)"
    problem.substitutions["JxB_pre"]      = "((Q*Pr)/(Ra*Pm))"
    problem.substitutions["inv_Pe_ff"]    = "(Ra*Pr)**(-1./2.)"


    #Equations
    problem.add_equation("T_z - dz(T) = 0")
    problem.add_equation("Oy - (dz(u) - dx(w)) = 0")
#    problem.add_equation("Ox - (dy(w) - dz(v)) = 0")
#    problem.add_equation("Jx - (dz(Bx) - dx(Bz)) = 0")

    problem.add_equation("dx(u)  + dy(v)  + dz(w)  = 0")

    problem.add_equation("dt(T) + w*dzT0   - inv_Pe_ff*Lap(T, T_z)          = -UdotGrad(T, T_z)")

    problem.add_equation("dt(u)  + dx(p)   + inv_Re_ff*Kx - JxB_pre*Jy     = v*Oz - w*Oy + JxB_pre*(Jy*Bz - Jz*By)")
    problem.add_equation("dt(w)  + dz(p)   + inv_Re_ff*Kz              - T = u*Oy - v*Ox + JxB_pre*(Jx*By - Jy*Bx) ")


    if k == 0:
#        problem.add_equation("Bx_z - dz(Bx) = 0")
#        problem.add_equation("By_z - dz(By) = 0")
        problem.add_equation("Jz_z = 0")
        problem.add_equation("Bz   = 0")
        problem.add_equation("Bx = 0")
        problem.add_equation("By = 0")
#        problem.add_equation("dt(Bx) - dz(u) - inv_Rem_ff*dz(Bx_z) = Bz*dz(u) - w*dz(Bx)")
#        problem.add_equation("dt(By) - dz(v) - inv_Rem_ff*dz(By_z) = Bz*dz(v) - w*dz(By)")
    else:
        problem.add_equation("dx(Bx)  + dy(By)  + dz(Bz)  = 0")
        problem.add_equation("dt(Bz) + inv_Rem_ff*(dx(Jy) - dy(Jx))                    - dz(w)  = BdotGrad(w, dz(w)) - UdotGrad(Bz, dz(Bz))")#need to figure out nonliner terms
        problem.add_equation("Jz_z - dz(Jz) = 0")
        problem.add_equation("dt(Jz) - inv_Rem_ff*(dx(dx(Jz)) + dy(dy(Jz)) + dz(Jz_z)) - dz(Oz) = -Lap(u*By - v*Bx, dz(u*By - v*Bx)) + dz(Bx*Ox + By*Oy + Bz*Oz) - dz(u*Jx + v*Jy + w*Jz)") 
#    problem.add_equation("Jx_z - dz(Jx) = 0")
#    problem.add_equation("dt(Jx) - inv_Rem_ff*(dx(dx(Jx)) + dy(dy(Jx)) + dz(Jx_z)) - dz(Ox) = 0")#-Lap(u*By - v*Bx, dz(u*By - v*Bx)) + dz(Bx*Ox + By*Oy + Bz*Oz) - dz(u*Jx + v*Jy + w*Jz)") 

    bcs = ['T', 'u', 'w']
    if k != 0:
        if noHorizB_BCs:
            bcs += ['Bx', 'By']
        else:
            bcs += ['Bz', 'dz(Jz)']

    for bc in bcs:
        problem.add_bc(" left({}) = 0".format(bc))
        if bc == 'w' and k == 0:
            problem.add_bc("right(p) = 0")
        else:
            problem.add_bc("right({}) = 0".format(bc))
    return problem

if __name__ == '__main__':
    Q = 1
    Ra = 3000
    nzs = [16, 32]
    k = 0

    problems = []
    for nz in nzs:
        problems.append(mhd_rbc_evp(Q, Ra, k, Nz=nz, noHorizB_BCs=True))
        problems.append(mhd_rbc_evp(Q, Ra, k, Nz=nz, noHorizB_BCs=False))

    evalues = []
    for i, p in enumerate(problems):
        solver = p.build_solver()
        solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)
        print('solve {} condition number: {:.2e}'.format(i, np.linalg.cond(solver.pencils[0].L_exp.A)))
        lamb = solver.eigenvalues
        lamb.imag[lamb.imag == 0] = 1e-16
        evalues.append(lamb)


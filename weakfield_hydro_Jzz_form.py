import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import time
import dedalus.public as de
import numpy as np
import os

comm = MPI.COMM_WORLD

def weakfield_hydro(Ra, k, Nz=24, noHorizB_BCs=True, kx=True):
    z = de.Chebyshev('z',Nz, interval=(0, 1))
    d = de.Domain([z],comm=MPI.COMM_SELF)

    variables=['p','u','v','w','Ox','Oy', 'Bx','By', 'Bz', 'Jz_z']
    problem = de.EVP(d,variables, eigenvalue='omega')

    #Parameters
    if kx:
        problem.parameters['kx']   = k
        problem.parameters['ky']   = 0
    else:
        problem.parameters['kx']   = 0
        problem.parameters['ky']   = k
    problem.parameters['Ra']   = Ra #Rayleigh number
    problem.parameters['Pm']   = 1   #Magnetic Prandtl number
    problem.parameters['dzT0'] = -1

    #Substitutions
    problem.substitutions["dt(A)"] = "omega*A"
    problem.substitutions["dx(A)"] = "1j*kx*A"
    problem.substitutions["dy(A)"] = "1j*ky*A"
    problem.substitutions['UdotGrad(A,A_z)'] = '(u*dx(A) + v*dy(A) + w*(A_z))'
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

    problem.substitutions['B0'] = '0'
    problem.substitutions['T'] = '0'

    #Equations
    problem.add_equation("dx(u)  + dy(v)  + dz(w)  = 0")

    problem.add_equation("dt(u)  + dx(p)   + inv_Re_ff*(dy(Oz)-dz(Oy)) = 0 ")
    problem.add_equation("dt(v)  + dy(p)   + inv_Re_ff*(dz(Ox)-dx(Oz)) = 0 ")
    problem.add_equation("dt(w)  + dz(p)   + inv_Re_ff*(dx(Oy)-dy(Ox)) = 0 ")

    problem.add_equation("Ox - (dy(w) - dz(v)) = 0")
    problem.add_equation("Oy - (dz(u) - dx(w)) = 0")


    problem.add_equation("dx(Bx)  + dy(By)  + dz(Bz)  = 0")

    problem.add_equation("dt(Bz) + inv_Rem_ff*(dx(Jy) - dy(Jx))                    = 0")#need to figure out nonliner terms
    problem.add_equation("dt(Jz) - inv_Rem_ff*(dx(dx(Jz)) + dy(dy(Jz)) + dz(Jz_z)) = 0") 

    problem.add_equation("Jz_z - dz(Jz) = 0")

    if noHorizB_BCs:
        bcs = ['Oy',  'Ox',  'w', 'Bx', 'By']
    else:
        bcs = ['Oy',  'Ox',  'w', 'Bz', 'Jz_z']

    for bc in bcs:
        problem.add_bc(" left({}) = 0".format(bc))
        problem.add_bc("right({}) = 0".format(bc))
    return problem

if __name__ == '__main__':
    Ra = 3000
    nzs = [16, 32, 64]

    problems = []
    for nz in nzs:
        problems.append(weakfield_hydro(Ra, 1, Nz=nz, kx=True,  noHorizB_BCs=True))
        problems.append(weakfield_hydro(Ra, 1, Nz=nz, kx=True,  noHorizB_BCs=False))
        problems.append(weakfield_hydro(Ra, 1, Nz=nz, kx=False, noHorizB_BCs=True))
        problems.append(weakfield_hydro(Ra, 1, Nz=nz, kx=False, noHorizB_BCs=False))

    evalues = []
    for i, p in enumerate(problems):
        solver = p.build_solver()
        solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)
        print('solve {} condition number: {:.2e}'.format(i, np.linalg.cond(solver.pencils[0].L_exp.A)))
        lamb = solver.eigenvalues
        lamb.imag[lamb.imag == 0] = 1e-16
        evalues.append(lamb)


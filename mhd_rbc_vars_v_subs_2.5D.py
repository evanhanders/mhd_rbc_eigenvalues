import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import time
import dedalus.public as de
import numpy as np
import os

comm = MPI.COMM_WORLD

def mhd_rbc_evp(Q, Ra, k, Nz=24, variables_form=True):
    z = de.Chebyshev('z',Nz, interval=(0, 1))
    d = de.Domain([z],comm=MPI.COMM_SELF)

    if variables_form:
        variables=['T','Tz','u','w','phi','Ax','Ay','Az','Bx','By', 'Bz','Jx','Jy','Oy', 'p']
    else:
        variables=['T','Tz','u','w','Ax','Ay','Az','Bx','By','Oy','p', 'phi']
    problem = de.EVP(d,variables, eigenvalue='omega')

    #Parameters
    problem.parameters['kx']   = k #horizontal wavenumber (x)
    problem.parameters['ky']   = 0 #horizontal wavenumber (y)
    problem.parameters['Ra']   = Ra #Rayleigh number
    problem.parameters['Pr']   = 1   #Prandtl number
    problem.parameters['Pm']   = 1   #Magnetic Prandtl number
    problem.parameters['Q']    = Q 
    problem.parameters['dzT0'] = -1

    #Substitutions
    problem.substitutions['v'] = '0'
    problem.substitutions['Ox'] = '0'
    problem.substitutions["dt(A)"] = "omega*A"
    problem.substitutions["dx(A)"] = "1j*kx*A"
    problem.substitutions["dy(A)"] = "1j*ky*A"
    problem.substitutions['UdotGrad(A,A_z)'] = '(u*dx(A) + v*dy(A) + w*(A_z))'
    problem.substitutions['Lap(A,A_z)'] = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
    if not variables_form:
        problem.substitutions["Bz"] = "dx(Ay)-dy(Ax)"
        problem.substitutions["Jx"] = "dy(Bz)-dz(By)"
        problem.substitutions["Jy"] = "dz(Bx)-dx(Bz)"
    problem.substitutions["Jz"] = "dx(By)-dy(Bx)"
    problem.substitutions["Kz"] = "dx(Oy)-dy(Ox)"
    problem.substitutions["Oz"] = "dx(v)-dy(u)"
    problem.substitutions["Ky"] = "dz(Ox)-dx(Oz)"
    problem.substitutions["Kx"] = "dy(Oz)-dz(Oy)"


    #Dimensionless parameter substitutions
    problem.substitutions["inv_Re_ff"]    = "(Pr/Ra)**(1./2.)"
    problem.substitutions["inv_Rem_ff"]   = "(inv_Re_ff / Pm)"
    problem.substitutions["JxB_pre"]      = "((Q*Pr)/(Ra*Pm))"
    problem.substitutions["inv_Pe_ff"]    = "(Ra*Pr)**(-1./2.)"

    #Equations
    problem.add_equation("Tz - dz(T) = 0")
    problem.add_equation("dt(T) + w*dzT0   - inv_Pe_ff*Lap(T, Tz)          = -UdotGrad(T, Tz)")

    problem.add_equation("dt(u)  + dx(p)   + inv_Re_ff*Kx - JxB_pre*Jy     = v*Oz - w*Oy + JxB_pre*(Jy*Bz - Jz*By)")
    problem.add_equation("dt(w)  + dz(p)   + inv_Re_ff*Kz              - T = u*Oy - v*Ox + JxB_pre*(Jx*By - Jy*Bx) ")

    problem.add_equation("dt(Ax) + dx(phi) + inv_Rem_ff*Jx - v             = v*Bz - w*By")
    problem.add_equation("dt(Ay) + dy(phi) + inv_Rem_ff*Jy + u             = w*Bx - u*Bz")
    problem.add_equation("dt(Az) + dz(phi) + inv_Rem_ff*Jz                 = u*By - v*Bx")

    problem.add_equation("dx(u)  + dy(v)  + dz(w)  = 0")
    problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0") #do I need dy here??

    problem.add_equation("Bx - (dy(Az) - dz(Ay)) = 0")
    problem.add_equation("By - (dz(Ax) - dx(Az)) = 0")
    if variables_form:
        problem.add_equation("Bz - (dx(Ay) - dy(Ax)) = 0")
        problem.add_equation("Jx - (dy(Bz) - dz(By)) = 0")
        problem.add_equation("Jy - (dz(Bx) - dx(Bz)) = 0")
    problem.add_equation("Oy - (dz(u) - dx(w)) = 0")

    bcs = ['T', 'u',  'w', 'Jx', 'Jy', 'Bz']
    for bc in bcs:
        problem.add_bc(" left({}) = 0".format(bc))
        problem.add_bc("right({}) = 0".format(bc))
    return problem

if __name__ == '__main__':
    Q  = 1
    Ra = 3000
    k  = 2
    nzs = [16, 32, 64]

    problems = []
    for nz in nzs:
        problems.append(mhd_rbc_evp(Q, Ra, k, Nz=nz, variables_form=True))
        problems.append(mhd_rbc_evp(Q, Ra, k, Nz=nz, variables_form=False))

    evalues = []
    for i, p in enumerate(problems):
        solver = p.build_solver()
        solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)
        print('solve {} condition number: {:.2e}'.format(i, np.linalg.cond(solver.pencils[0].L_exp.A)))
        lamb = solver.eigenvalues
        evalues.append(lamb)

    # Fig 1 & 2 : Variables vs subs (lowres & Hires)
    fig = plt.figure()
    for j in range(3):
        colors = ['blue', 'orange']
        label  = ['vars', 'subs']
        for i in range(2):
            evals = evalues[i + 2*j]
            plt.scatter(evals.real[evals.real >= 0],  np.abs(evals.imag[evals.real >= 0]), marker='o', c=colors[i], label=label[i], alpha=0.3)
            plt.scatter(-evals.real[evals.real < 0],  np.abs(evals.imag[evals.real < 0]), marker='+', c=colors[i], s=10, alpha=0.3)

        plt.scatter(1e99,  1e99, marker='o', c='k', label='real = positive', alpha=0.3)
        plt.scatter(1e99,  1e99, marker='+', c='k', label='real = negative', s=10, alpha=0.3)
        plt.legend()

        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(1e-20, 1e8)
        plt.xlim(1e-2, 1e8)
        plt.ylabel('abs(imag)')
        plt.xlabel('real')

        fig.savefig('mhd2.5_vars_vs_subs_nz{}_ra{}_k{}.png'.format(nzs[j], Ra, k), bbox_inches='tight', dpi=300)
        plt.clf()

    # Fig 3 & 4 : Hires vs Lowres 
    fig = plt.figure()
    file_label = ['variables', 'substitutions']
    for j in range(len(file_label)):
        colors = ['indigo', 'green', 'orange']
        label  = ['nz={}'.format(nz) for nz in nzs]
        for i in range(len(nzs)):
            evals = evalues[j + 2*i]
            plt.scatter(evals.real[evals.real >= 0],  np.abs(evals.imag[evals.real >= 0]), marker='o', c=colors[i], label=label[i], alpha=0.3)
            plt.scatter(-evals.real[evals.real < 0],  np.abs(evals.imag[evals.real < 0]), marker='+', c=colors[i], s=10, alpha=0.3)

        plt.scatter(1e99,  1e99, marker='o', c='k', label='real = positive', alpha=0.3)
        plt.scatter(1e99,  1e99, marker='+', c='k', label='real = negative', s=10, alpha=0.3)
        plt.legend()

        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(1e-20, 1e8)
        plt.xlim(1e-2, 1e8)
        plt.ylabel('abs(imag)')
        plt.xlabel('real')

        fig.savefig('mhd2.5_{}_low_hi_res_ra{}_k{}.png'.format(file_label[j], Ra, k), bbox_inches='tight', dpi=300)
        plt.clf()





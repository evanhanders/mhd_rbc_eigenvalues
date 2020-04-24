import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
import time
import dedalus.public as de
import numpy as np
import os

comm = MPI.COMM_WORLD

def rbc_evp(Ra, k, Nz=24):
    z = de.Chebyshev('z',Nz, interval=(0, 1))
    d = de.Domain([z],comm=MPI.COMM_SELF)

    variables=['T','Tz','p','u','v','w','Ox','Oy']
    problem = de.EVP(d,variables, eigenvalue='omega')

    #Parameters
    problem.parameters['kx']   = k/np.sqrt(2) #horizontal wavenumber (x)
    problem.parameters['ky']   = k/np.sqrt(2) #horizontal wavenumber (y)
    problem.parameters['Ra']   = Ra #Rayleigh number
    problem.parameters['Pr']   = 1   #Prandtl number
    problem.parameters['Pm']   = 1   #Magnetic Prandtl number
    problem.parameters['dzT0'] = -1

    #Substitutions
    problem.substitutions["dt(A)"] = "omega*A"
    problem.substitutions["dx(A)"] = "1j*kx*A"
    problem.substitutions["dy(A)"] = "1j*ky*A"
    problem.substitutions['UdotGrad(A,A_z)'] = '(u*dx(A) + v*dy(A) + w*(A_z))'
    problem.substitutions['Lap(A,A_z)'] = '(dx(dx(A)) + dy(dy(A)) + dz(A_z))'
    problem.substitutions["Oz"] = "dx(v)-dy(u)"
    problem.substitutions["Kz"] = "dx(Oy)-dy(Ox)"
    problem.substitutions["Ky"] = "dz(Ox)-dx(Oz)"
    problem.substitutions["Kx"] = "dy(Oz)-dz(Oy)"


    #Dimensionless parameter substitutions
    problem.substitutions["inv_Re_ff"]    = "(Pr/Ra)**(1./2.)"
    problem.substitutions["inv_Pe_ff"]    = "(Ra*Pr)**(-1./2.)"

    #Equations
    problem.add_equation("dt(T) + w*dzT0   - inv_Pe_ff*Lap(T, Tz)          = -UdotGrad(T, Tz)")

    problem.add_equation("dt(u)  + dx(p)   + inv_Re_ff*Kx     = v*Oz - w*Oy")
    problem.add_equation("dt(v)  + dy(p)   + inv_Re_ff*Ky     = w*Ox - u*Oz")
    problem.add_equation("dt(w)  + dz(p)   + inv_Re_ff*Kz - T = u*Oy - v*Ox")

    problem.add_equation("dx(u)  + dy(v)  + dz(w)  = 0")

    problem.add_equation("Ox - (dy(w) - dz(v)) = 0")
    problem.add_equation("Oy - (dz(u) - dx(w)) = 0")
    problem.add_equation("Tz - dz(T) = 0")

    bcs = ['T', 'u',  'v',  'w']
    for bc in bcs:
        problem.add_bc(" left({}) = 0".format(bc))
        problem.add_bc("right({}) = 0".format(bc))
    return problem

if __name__ == '__main__':
    Ra = 1e4
    k  = 3
    nzs = [16, 32, 64]

    problems = []
    for nz in nzs:
        problems.append(rbc_evp(Ra, k, Nz=nz))

    evalues = []
    for i, p in enumerate(problems):
        solver = p.build_solver()
        solver.solve_dense(solver.pencils[0], rebuild_coeffs=True)
        print('solve {} condition number: {:.2e}'.format(i, np.linalg.cond(solver.pencils[0].L_exp.A)))
        lamb = solver.eigenvalues
        lamb = lamb[np.argsort(lamb.real)]
        evalues.append(lamb)

    # Fig 1 & 2 : Variables vs subs (lowres & Hires)
    fig = plt.figure()
    for j in range(3):
        evals = evalues[j]
        plt.scatter(evals.real[evals.real >= 0],  np.abs(evals.imag[evals.real >= 0]), marker='o', c='indigo', alpha=0.3)
        plt.scatter(-evals.real[evals.real < 0],  np.abs(evals.imag[evals.real < 0]), marker='+', c='indigo', s=10, alpha=0.3)

        plt.scatter(1e99,  1e99, marker='o', c='k', label='real = positive', alpha=0.3)
        plt.scatter(1e99,  1e99, marker='+', c='k', label='real = negative', s=10, alpha=0.3)
        plt.legend()

        plt.yscale('log')
        plt.xscale('log')
        plt.ylim(1e-40, 1e-8)
        plt.xlim(1e-2, 1e5)
        plt.ylabel('abs(imag)')
        plt.xlabel('real')

        fig.savefig('hydro_vars_vs_subs_nz{}_ra{}_k{}.png'.format(nzs[j], Ra, k), bbox_inches='tight', dpi=300)
        plt.clf()

    # Fig 3 & 4 : Hires vs Lowres 
    fig = plt.figure()
    colors = ['indigo', 'green', 'orange']
    label  = ['nz={}'.format(nz) for nz in nzs]
    for i in range(len(nzs)):
        evals = evalues[i]
        plt.scatter(evals.real[evals.real >= 0],  np.abs(evals.imag[evals.real >= 0]), marker='o', c=colors[i], label=label[i], alpha=0.3)
        plt.scatter(-evals.real[evals.real < 0],  np.abs(evals.imag[evals.real < 0]), marker='+', c=colors[i], s=10, alpha=0.3)

    plt.scatter(1e99,  1e99, marker='o', c='k', label='real = positive', alpha=0.3)
    plt.scatter(1e99,  1e99, marker='+', c='k', label='real = negative', s=10, alpha=0.3)
    plt.legend()

    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e-40, 1e-8)
    plt.xlim(1e-2, 1e5)
    plt.ylabel('abs(imag)')
    plt.xlabel('real')

    fig.savefig('hydro_low_hi_res_ra{}_k{}.png'.format(Ra, k), bbox_inches='tight', dpi=300)
    plt.clf()





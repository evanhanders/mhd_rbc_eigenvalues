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

    problem = de.EVP(d,['p', 'b', 'u', 'w', 'bz', 'uz', 'wz'], eigenvalue='omega')
    problem.parameters['k'] = k #horizontal wavenumber
    problem.parameters['Ra'] = Ra #Rayleigh number, rigid-rigid
    problem.parameters['Pr'] = 1  #Prandtl number
    problem.parameters['dzT0'] = 1
    problem.substitutions['dt(A)'] = 'omega*A'
    problem.substitutions['dx(A)'] = '1j*k*A'

    #Boussinesq eqns -- nondimensionalized on thermal diffusion timescale
    #Incompressibility
    problem.add_equation("dx(u) + wz = 0")
    #Momentum eqns
    problem.add_equation("dt(u) - Pr*(dx(dx(u)) + dz(uz)) + dx(p)           = -u*dx(u) - w*uz")
    problem.add_equation("dt(w) - Pr*(dx(dx(w)) + dz(wz)) + dz(p) - Ra*Pr*b = -u*dx(w) - w*wz")
    #Temp eqn
    problem.add_equation("dt(b) - w*dzT0 - (dx(dx(b)) + dz(bz)) = -u*dx(b) - w*bz")
    #Derivative defns
    problem.add_equation("dz(u) - uz = 0")
    problem.add_equation("dz(w) - wz = 0")
    problem.add_equation("dz(b) - bz = 0")

    bcs = ['b', 'u',  'w']
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
        print(lamb)
        lamb.imag[lamb.imag == 0] = 1e-16
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
        plt.ylim(1e-20, 1e0)
        plt.xlim(1e0, 1e5)
        plt.ylabel('abs(imag)')
        plt.xlabel('real')

        fig.savefig('hydro_plain_nz{}_ra{}_k{}.png'.format(nzs[j], Ra, k), bbox_inches='tight', dpi=300)
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
    plt.ylim(1e-20, 1e0)
    plt.xlim(1e0, 1e5)
    plt.ylabel('abs(imag)')
    plt.xlabel('real')

    fig.savefig('hydro_plain_low_hi_res_ra{}_k{}.png'.format(Ra, k), bbox_inches='tight', dpi=300)
    plt.clf()





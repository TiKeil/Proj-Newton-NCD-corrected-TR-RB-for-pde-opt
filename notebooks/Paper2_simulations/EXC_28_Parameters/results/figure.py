# ~~~
# This file is part of the paper:
#
#           "An adaptive projected Newton non-conforming dual approach
#         for trust-region reduced basis approximation of PDE-constrained
#                           parameter optimization"
#
#   https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt
#
# Copyright 2019-2020 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Luca Mechelli (2020)
#   Tim Keil      (2020)
# ~~~

import matplotlib.pyplot as plt
# import tikzplotlib
import sys
import numpy as np
path = '../../../'
sys.path.append(path)
from pdeopt.tools import get_data

directory = 'Starter4/'

mu_est = False
mu_error = True

colorclass0 =(0.65, 0.00, 0.15)
colorclass1 =(0.84, 0.19, 0.15)
colorclass2 =(0.99, 0.68, 0.38)
colorclass3 =(0.96, 0.43, 0.26)
colorclass4 = 'black'
colorclass5 =(0.17, 0.25, 0.98)

# I want to have these methods in my plot: 
method_tuple = [[1, 'FOM TR-Newton-CG'],
                [4, 'TR-RB Newton lag. '],
                [2, 'TR-RB Newton lag. optional '],
                [8, 'Keil et al. 2020'],
                [5, 'Keil et al. 2020 pr. New. '],
                [3, 'Alg. 1 Dir. Tay. Skip_enr. ']
                ]

if mu_est is False and mu_error is False:
    times_full_0 , J_error_0, FOC_0, _ = get_data(directory,method_tuple[0][0], FOC=True, j_list=False)
    times_full_1 , J_error_1, FOC_1, j_list_1 = get_data(directory,method_tuple[1][0], FOC=True)
    times_full_2 , J_error_2, FOC_2, j_list_2 = get_data(directory,method_tuple[2][0], FOC=True)
    times_full_3 , J_error_3, FOC_3, j_list_3 = get_data(directory,method_tuple[3][0], FOC=True)
    times_full_4 , J_error_4, FOC_4, j_list_4 = get_data(directory,method_tuple[4][0], FOC=True)
    times_full_5 , J_error_5, FOC_5, j_list_5 = get_data(directory,method_tuple[5][0], FOC=True)
elif mu_est is False and mu_error is True:
    times_full_0 , J_error_0 , mu_error_0 , FOC_0, _ = get_data(directory,method_tuple[0][0], mu_error_=mu_error, FOC=True, j_list=False)
    times_full_1 , J_error_1 , mu_error_1 , FOC_1, j_list_1 = get_data(directory,method_tuple[1][0], mu_error_=mu_error, FOC=True)
    times_full_2 , J_error_2 , mu_error_2 , FOC_2, j_list_2 = get_data(directory,method_tuple[2][0], mu_error_=mu_error, FOC=True)
    times_full_3 , J_error_3 , mu_error_3 , FOC_3, j_list_3 = get_data(directory,method_tuple[3][0], mu_error_=mu_error, FOC=True)
    times_full_4 , J_error_4 , mu_error_4 , FOC_4, j_list_4 = get_data(directory,method_tuple[4][0], mu_error_=mu_error, FOC=True)
    times_full_5 , J_error_5 , mu_error_5 , FOC_5, j_list_5 = get_data(directory,method_tuple[5][0], mu_error_=mu_error, FOC=True)
elif mu_est is True:
    times_full_0 , J_error_0 , mu_error_0 , times_mu_0 , mu_est_0, FOC_0, _ = get_data(directory,method_tuple[0][0], mu_est, mu_est, FOC=True, j_list=False)
    times_full_1 , J_error_1 , mu_error_1 , times_mu_1 , mu_est_1, FOC_1, j_list_1 = get_data(directory,method_tuple[1][0], mu_est, mu_est, FOC=True)
    times_full_2 , J_error_2 , mu_error_2 , times_mu_2 , mu_est_2, FOC_2, j_list_2 = get_data(directory,method_tuple[2][0], mu_est, mu_est, FOC=True)
    times_full_3 , J_error_3 , mu_error_3 , times_mu_3 , mu_est_3, FOC_3, j_list_3 = get_data(directory,method_tuple[3][0], mu_est, mu_est, FOC=True)
    times_full_4 , J_error_4 , mu_error_4 , times_mu_4 , mu_est_4, FOC_4, j_list_4 = get_data(directory,method_tuple[4][0], mu_est, mu_est, FOC=True)
    times_full_5 , J_error_5 , mu_error_5 , times_mu_5 , mu_est_5, FOC_5, j_list_5 = get_data(directory,method_tuple[5][0], mu_est, mu_est, FOC=True)
    #fix mu_est
    times_mu_0 = [ti + times_full_0[0] for ti in times_mu_0]
    times_mu_1 = [ti + times_full_1[0] for ti in times_mu_1]
    times_mu_2 = [ti + times_full_2[0] for ti in times_mu_2]
    times_mu_3 = [ti + times_full_3[0] for ti in times_mu_3]
    times_mu_4 = [ti + times_full_4[0] for ti in times_mu_4]
    times_mu_5 = [ti + times_full_5[0] for ti in times_mu_5]
if 1:
    timings_figure = plt.figure(figsize=(10,5))
    plt.semilogy(times_full_0 ,J_error_0 , '-', color=colorclass0, marker='^', label=method_tuple[0][1])
    plt.semilogy(times_full_1 ,J_error_1 , '-', color=colorclass1, marker='v', label=method_tuple[1][1])
    plt.semilogy(times_full_2 ,J_error_2 , '-', color=colorclass2, marker='x', label=method_tuple[2][1])
    plt.semilogy(times_full_3 ,J_error_3 , '-', color=colorclass3, marker='p', label=method_tuple[3][1])
    plt.semilogy(times_full_4 ,J_error_4 , '-', color=colorclass4, marker='o', label=method_tuple[4][1])
    plt.semilogy(times_full_5 ,J_error_5 , '-', color=colorclass5, marker='D', label=method_tuple[5][1])
    # plt.xlim([-3,3600])
    # plt.ylim([1e-18, 1e4])
    plt.xlabel('time in seconds [s]',fontsize=14)
    plt.ylabel('$| \hat{\mathcal{J}}_h(\overline{\mu})-\hat{\mathcal{J}}^k_n(\mu_k) |$', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlim([-1,30])
    plt.grid()
    plt.legend(fontsize=10)
    # plt.legend(loc='lower center', fontsize=10)

    # tikzplotlib.save("{}J_error.tex".format(directory))
    # timings_figure.savefig('{}J_error_plot.pdf'.format(directory), format='pdf', bbox_inches="tight")

if 1:
    timings_figure_3 = plt.figure(figsize=(10,5))
    plt.semilogy(times_full_0, FOC_0 , '-', color=colorclass0 , marker='^', label=method_tuple[0][1])
    plt.semilogy(times_full_1, FOC_1 , '-', color=colorclass1 , marker='v', label=method_tuple[1][1])
    plt.semilogy(times_full_2, FOC_2 , '-', color=colorclass2 , marker='o', label=method_tuple[2][1])
    plt.semilogy(times_full_3, FOC_3 , '-', color=colorclass3 , marker='p', label=method_tuple[3][1])
    plt.semilogy(times_full_4, FOC_4 , '-', color=colorclass4 , marker='x', label=method_tuple[4][1])
    plt.semilogy(times_full_5, FOC_5 , '-', color=colorclass5 , marker='D', label=method_tuple[5][1])

    # plt.xlim([-3,3600])
    # plt.ylim([1e-18, 1e4])
    plt.xlabel('time in seconds [s]',fontsize=14)
    plt.ylabel('FOC condition', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.xlim([-1,30])
    plt.grid()
    # plt.legend(loc='lower center', fontsize=10)
    plt.legend(fontsize=10)

    # tikzplotlib.save("{}FOC.tex".format(directory))
    # timings_figure_3.savefig('{}FOC.pdf'.format(directory), format='pdf', bbox_inches="tight")

if 1:
    timings_figure_3 = plt.figure(figsize=(10,10))
    iterations_1 = np.arange(len(j_list_1))
    iterations_2 = np.arange(len(j_list_2))
    iterations_3 = np.arange(len(j_list_3))
    iterations_4 = np.arange(len(j_list_4))
    iterations_5 = np.arange(len(j_list_5))
    #plt.plot(iterations_1, j_list_1 , '-', color=colorclass1 , marker='v', label=method_tuple[1][1])
    #plt.plot(iterations_2, j_list_2 , '-', color=colorclass2 , marker='x', label=method_tuple[2][1])
    plt.plot(iterations_3, j_list_3 , '-', color=colorclass3 , marker='p', label=method_tuple[3][1], linewidth= 2)
    plt.plot(iterations_4, j_list_4 , '-', color=colorclass4 , marker='o', label=method_tuple[4][1], linewidth= 2)
    plt.plot(iterations_5, j_list_5 , '-', color=colorclass5 , marker='D', label=method_tuple[5][1], linewidth= 2)

    # plt.xlim([-3,3600])
    # plt.ylim([1e-18, 1e4])
    plt.xlabel('Outer iteration k',fontsize=28)
    #plt.ylabel('FOC condition', fontsize=14)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # plt.xlim([-1,30])
    plt.grid()
    # plt.legend(loc='lower center', fontsize=10)
    #plt.legend(fontsize=30)

    # tikzplotlib.save("{}FOC.tex".format(directory))
    # timings_figure_3.savefig('{}FOC.pdf'.format(directory), format='pdf', bbox_inches="tight")

if mu_error is True:
    timings_figure = plt.figure(figsize=(10,10))
    #plt.semilogy(times_full_0 ,mu_error_0 , '-', color=colorclass0 , marker='^', label=method_tuple[0][1])
    #plt.semilogy(times_full_1 ,mu_error_1 , '-', color=colorclass1 , marker='v', label=method_tuple[1][1])
    #plt.semilogy(times_full_2 ,mu_error_2 , '-', color=colorclass2 , marker='x', label=method_tuple[2][1])
    plt.semilogy(times_full_3 ,mu_error_3 , '-', color=colorclass3 , marker='p', label=method_tuple[3][1], linewidth= 2)
    plt.semilogy(times_full_4 ,mu_error_4 , '-', color=colorclass4 , marker='o', label=method_tuple[4][1], linewidth= 2)
    plt.semilogy(times_full_5 ,mu_error_5 , '-', color=colorclass5 , marker='D', label=method_tuple[5][1], linewidth= 2)
    plt.xlabel('Time [s]',fontsize=28)
    #plt.ylabel('$\| \overline{\mu}-\mu_k \|$', fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlim((0,300))
    plt.ylim((1e-7,1e5))
    plt.grid()
    plt.legend(fontsize=30, bbox_to_anchor= [1.025,1.025], loc= 'upper right')

    # tikzplotlib.save("{}mu_error.tex".format(directory))
    # timings_figure.savefig('{}mu_error_plot.pdf'.format(directory), format='pdf', bbox_inches="tight")

plt.show()

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
#   Luca Mechelli (2019 - 2020)
#   Tim Keil      (2019 - 2020)
# ~~~

import numpy as np
import time
from copy import deepcopy

from pymor.algorithms.gram_schmidt import gram_schmidt
from pdeopt.reductor import QuadraticPdeoptStationaryCoerciveReductor
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres as GMRESScipy
from scipy.sparse.linalg import minres as MINRESScipy
from pdeopt.tools import truncated_conj_grad as TruncCG
from pdeopt.tools import truncated_conj_grad_Steihaug as TruncCGSteihaug
from pdeopt.tools import truncated_stabilzed_biconj_grad as TruncBICGStab
from scipy.sparse.linalg import bicgstab as BICGScipy


def projection_onto_range(parameter_space, mu):

    ranges = parameter_space.ranges
    new_mu = []
    for (key, size) in sorted(parameter_space.parameters.items()):
        range_ = ranges[key]
        for j in range(size):
            if mu[key][j] < range_[0]:
                new_mu.append(range_[0])
            elif mu[key][j] > range_[1]:
                new_mu.append(range_[1])
            else:
                new_mu.append(mu[key][j])
    return parameter_space.parameters.parse(new_mu)

def active_and_inactive_sets(parameter_space, mu, epsilon):

    Act = []

    ranges = parameter_space.ranges
    for (key,size) in sorted(parameter_space.parameters.items()):
        range_ = ranges[key]
        for j in range(size):
            if mu[key][j] - range_[0] <= epsilon:
                Act.append(1.0)
            elif range_[1] - mu[key][j] <= epsilon:
                Act.append(1.0)
            else:
                Act.append(0.0)

    Act = np.array(Act)
    Inact = np.ones(Act.shape) - Act

    return Act, Inact


def armijo_rule(opt_model, parameter_space, TR_parameters, mu_i, Ji, direction):
    j = 0
    condition = True
    while condition and j < TR_parameters['max_iterations_armijo']:
        mu_ip1 = mu_i + (TR_parameters['initial_step_armijo'] ** j) * direction
        mu_ip1_dict = opt_model.primal_model.parameters.parse(mu_ip1)
        mu_ip1_dict = projection_onto_range(parameter_space,mu_ip1_dict)
        mu_ip1 = mu_ip1_dict.to_numpy()
        Jip1 = opt_model.output_functional_hat(mu_ip1_dict)

        if not TR_parameters['full_order_model']:
            u_cp = opt_model.solve(mu_ip1_dict)
            p_cp = opt_model.solve_dual(mu_ip1_dict)
            est = opt_model.estimate_output_functional_hat(u_cp, p_cp, mu_ip1_dict)
        else:
            est = 0.0

        if  Jip1 <= Ji - (TR_parameters['armijo_alpha'] / ((TR_parameters['initial_step_armijo'] ** j))) * (np.linalg.norm(mu_ip1-mu_i)**2) and abs(est / Jip1) <= TR_parameters['radius']:
            condition = False

        j = j + 1

    if condition:  # This means that we exit the loop because of maximum iteration reached
        print("Maximum iteration for Armijo rule reached")
        mu_ip1 = mu_i
        mu_ip1_dict = opt_model.primal_model.parameters.parse(mu_ip1)
        Jip1 = Ji
        est = TR_parameters['radius']*Ji # so that the Qian-Grepl stop as well


    return mu_ip1, mu_ip1_dict, Jip1, abs(est / Jip1) #the last is needed for the boundary criterium

def compute_new_hessian_approximation(new_mu,old_mu,new_gradient,old_gradient,old_B):

    gk = new_gradient-old_gradient
    pk = new_mu-old_mu

    den = gk.dot(pk)

    if den>0.0:
        Hkgk = old_B.dot(gk)
        coeff = gk.dot(Hkgk)

        Hkgkpkt = np.outer(Hkgk,pk)

        pkHkgkt = np.outer(pk,Hkgk)

        pkpkt = np.outer(pk,pk)

        new_B = old_B + (den+coeff)/(den*den) * pkpkt - (1.0/den) * Hkgkpkt - (1.0/den)*pkHkgkt
    else:
        print("Curvature condition: {}".format(den))
        print("Reset direction to - gradient")
        new_B = np.eye(old_gradient.size)

    return new_B

def compute_modified_hessian_action_matrix_version(H,Active,Inactive,eta):

    etaA = np.multiply(Active, eta)
    etaI = np.multiply(Inactive, eta)

    Hessian_prod = H.dot(etaI)
    Action_of_modified_H = etaA + np.multiply(Inactive, Hessian_prod)

    return Action_of_modified_H


def solve_optimization_subproblem_BFGS(opt_model, parameter_space, mu_k_dict, TR_parameters, timing=False):
    #This is used by the TR algorithm in the paper Keil et al. '20,  with which we compare our new proposed method
    print('___ starting subproblem')
    if 'beta' not in TR_parameters:
        print('Setting beta to the default 0.95')
        TR_parameters['beta'] = 0.95


    tic_ = time.time()
    times = []
    mus = []
    Js = []
    FOCs = []

    mu_diff = 1e6
    J_diff = 1e6
    Ji = opt_model.output_functional_hat(mu_k_dict)

    gradient = opt_model.output_functional_hat_gradient(mu_k_dict)
    normgrad = np.linalg.norm(gradient)
    mu_i = mu_k_dict.to_numpy()
    mu_i_dict = opt_model.primal_model.parameters.parse(mu_i)

    mu_i_1 = mu_i - gradient
    mu_i_1_dict = projection_onto_range(parameter_space, opt_model.primal_model.parameters.parse(mu_i_1))
    mu_i_1 = mu_i_1_dict.to_numpy()
    epsilon_i = TR_parameters['epsilon_i']
    if not isinstance(epsilon_i,float):
        epsilon_i = np.linalg.norm(mu_i_1 - mu_i)#/(np.linalg.norm(mu_i)+1e-8)
    B = np.eye(mu_i.size)
    Active_i, Inactive_i = active_and_inactive_sets(parameter_space, mu_i_dict, epsilon_i)

    i = 0
    while i < TR_parameters['max_iterations_subproblem']:
        if i>0:
            if boundary_TR_criterium>= TR_parameters['beta']*TR_parameters['radius']:
                print('boundary criterium of the TR satisfied, so stopping the sub-problem solver')

                return mu_ip1_dict, Jcp, i, Jip1, FOCs
            if normgrad < TR_parameters['sub_tolerance'] or J_diff < TR_parameters['safety_tolerance'] or mu_diff< TR_parameters['safety_tolerance']:
                print("Subproblem converged: FOC = {}, mu_diff = {}, J_diff = {} ".format(normgrad,mu_diff,J_diff))
                break

        if i == 0:
            print("Computing the approximate Cauchy point and then start the BFGS method")
            direction = -gradient
        else:
            if Inactive_i.sum() == 0.0:
                direction = -gradient
            else:
                direction = compute_modified_hessian_action_matrix_version(B, Active_i, Inactive_i, -gradient)
            if np.dot(direction,gradient) >= -1e-14:
                print('Not a descendent direction ... taking -gradient as direction')
                direction = -gradient

        mu_ip1, mu_ip1_dict, Jip1, boundary_TR_criterium = armijo_rule(opt_model, parameter_space, TR_parameters, mu_i, Ji, direction)

        if i == 0:
            Jcp = Jip1


        mu_diff = np.linalg.norm(mu_i - mu_ip1) / np.linalg.norm(mu_i)
        J_diff = abs(Ji - Jip1) / abs(Ji)
        old_mu = deepcopy(mu_i)
        mu_i_dict = mu_ip1_dict
        Ji = Jip1

        old_gradient = deepcopy(gradient)
        gradient = opt_model.output_functional_hat_gradient(mu_i_dict)
        mu_box = opt_model.primal_model.parameters.parse(mu_i_dict.to_numpy()-gradient)
        first_order_criticity = mu_i_dict.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
        normgrad = np.linalg.norm(first_order_criticity)
        mu_i = mu_i_dict.to_numpy()
        mu_i_dict = opt_model.primal_model.parameters.parse(mu_i)

        mu_i_1 = mu_i - gradient
        mu_i_1_dict = projection_onto_range(parameter_space,opt_model.primal_model.parameters.parse(mu_i_1))
        mu_i_1 = mu_i_1_dict.to_numpy()
        if not isinstance(epsilon_i,float):
            epsilon_i = np.linalg.norm(mu_i_1 - mu_i)#/(np.linalg.norm(mu_i)+1e-8)
        Active_i, Inactive_i = active_and_inactive_sets(parameter_space, mu_i_dict, epsilon_i)
        B = compute_new_hessian_approximation(mu_i, old_mu, gradient, old_gradient, B)


        times.append(time.time() -tic_)
        mus.append(mu_ip1)
        Js.append(Ji)
        FOCs.append(normgrad)
        i = i + 1

    print("relative differences mu {} and J {}".format(mu_diff, J_diff))

    if timing:
        return mu_ip1_dict, Jcp, i, Jip1, times, mus, Js, FOCs
    else:
        return mu_ip1_dict, Jcp, i, Jip1, FOCs


def modified_hessian_action(mu,Active,Inactive,opt_model,eta):

    etaA = np.multiply(Active,eta)
    etaI = np.multiply(Inactive,eta)

    Action_on_I = opt_model.output_functional_hessian_operator(mu, etaI)

    Action_of_modified_operator = etaA + np.multiply(Inactive,Action_on_I)

    return Action_of_modified_operator


def solve_optimization_subproblem_NewtonMethod(opt_model, parameter_space, mu_k_dict, TR_parameters, timing=False):
    print('___ starting subproblem')
    if 'beta' not in TR_parameters:
        print('Setting beta to the default 0.95')
        TR_parameters['beta'] = 0.95

    if 'iterative_solver' not in TR_parameters:
        TR_parameters['iterative_solver'] = 'CG'

    tic_toc = time.time()
    times = []
    mus = []
    Js = []
    FOCs = []
    times_est_evaluations = []
    mu_est = []
    additional_criteria = 0

    mu_diff = 1
    J_diff = 1
    Ji = opt_model.output_functional_hat(mu_k_dict)

    gradient = opt_model.output_functional_hat_gradient(mu_k_dict)
    normgrad = np.linalg.norm(gradient)
    mu_i = mu_k_dict.to_numpy()
    mu_i_dict = opt_model.primal_model.parameters.parse(mu_i)

    mu_i_1 = mu_i - gradient
    mu_i_1_dict = projection_onto_range(parameter_space, opt_model.primal_model.parameters.parse(mu_i_1))
    mu_i_1 = mu_i_1_dict.to_numpy()
    epsilon_i = TR_parameters['epsilon_i']
    if not isinstance(epsilon_i,float):
        epsilon_i = np.linalg.norm(mu_i_1 - mu_i)#/(np.linalg.norm(mu_i)+1e-8)

    i = 0
    old_gradient = 1
    while i < TR_parameters['max_iterations_subproblem']:
        if i>0:
            if boundary_TR_criterium>= TR_parameters['beta']*TR_parameters['radius']:
                print('boundary criterium of the TR satisfied, so stopping the sub-problem {} {}'.format(boundary_TR_criterium,normgrad))
                return mu_ip1_dict, Jcp, i, Jip1, FOCs
            if normgrad < TR_parameters['sub_tolerance'] or J_diff < TR_parameters['safety_tolerance'] or mu_diff< TR_parameters['safety_tolerance']:
                additional_criteria = 1
                print("Subproblem converged: mu_diff = {}, J_diff = {}, FOC = {}".format(mu_diff,J_diff,normgrad))
            if additional_criteria:
                break

        if i == 0:
            print("Computing the approximate Cauchy point and then start the Newton method")
            deltamu = gradient

        else:
            Active_i, Inactive_i = active_and_inactive_sets(parameter_space, mu_i_dict, epsilon_i)

            if Inactive_i.sum() == 0.0:
                deltamu = gradient
            else:
                print("Using CG for the linear system")
                deltamu, itcg,rescg, infocg = TruncCG(A_func=lambda v: modified_hessian_action(mu=mu_i_dict, Active= Active_i, Inactive= Inactive_i, opt_model=opt_model, eta=v), b= gradient, tol = 1.e-10)
                if infocg > 0:
                    print("Choosing the gradient as direction")
                    deltamu = gradient
            if np.dot(-deltamu,gradient) >= -1.e-14:
                print('Not a descendent direction ... taking gradient as direction')
                deltamu = gradient

        mu_ip1, mu_ip1_dict, Jip1, boundary_TR_criterium = armijo_rule(opt_model, parameter_space, TR_parameters, mu_i, Ji, -deltamu)


        if i == 0:
            Jcp = Jip1

        mu_diff = np.linalg.norm(mu_i - mu_ip1) / np.linalg.norm(mu_i)
        J_diff = abs(Ji - Jip1) / abs(Ji)
        mu_i_dict = mu_ip1_dict
        Ji = Jip1

        gradient = opt_model.output_functional_hat_gradient(mu_i_dict)
        mu_box = opt_model.primal_model.parameters.parse(mu_i_dict.to_numpy()-gradient)
        first_order_criticity = mu_i_dict.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
        normgrad = np.linalg.norm(first_order_criticity)

        mu_i = mu_i_dict.to_numpy()
        mu_i_dict = opt_model.primal_model.parameters.parse(mu_i)

        mu_i_1 = mu_i - gradient
        mu_i_1_dict = projection_onto_range(parameter_space,opt_model.primal_model.parameters.parse(mu_i_1))
        mu_i_1 = mu_i_1_dict.to_numpy()
        if not isinstance(epsilon_i,float):
            epsilon_i = np.linalg.norm(mu_i_1 - mu_i)#/(np.linalg.norm(mu_i)+1e-8)

        times.append(time.time() -tic_toc)
        mus.append(mu_ip1)
        Js.append(Ji)
        FOCs.append(normgrad)

        i = i + 1

    print("relative differences mu {} and J {}".format(mu_diff, J_diff))

    if timing:
        return mu_ip1_dict, Jcp, i, Jip1, times, mus, Js, FOCs
    else:
        return mu_ip1_dict, Jcp, i, Jip1, FOCs


def enrichment_step(mu, reductor, adaptive_taylor=False, U = None, P = None):

    if adaptive_taylor:
        new_reductor = deepcopy(reductor)
        u, p = new_reductor.extend_adaptive_taylor(mu, U = U, P = P)
        opt_rom = new_reductor.reduce()
        return opt_rom, new_reductor, u, p
    else:
        new_reductor = deepcopy(reductor)
        u, p = new_reductor.extend_bases(mu, U = U, P = P)
        opt_rom = new_reductor.reduce()
        return opt_rom, new_reductor, u, p


def TR_algorithm(opt_rom, reductor, parameter_space, TR_parameters=None, extension_params=None, opt_fom=None, return_opt_rom=False):
    if TR_parameters is None:
        mu_k = parameter_space.sample_randomly(1)[0]
        TR_parameters = {'radius': 0.1, 'sub_tolerance': 1e-8, 'max_iterations': 40, 'max_iterations_subproblem': 400,
                         'starting_parameter': mu_k, 'max_iterations_armijo': 50, 'initial_step_armijo': 0.5, 'armijo_alpha': 1e-4,
                         'opt_method': 'BFGSMethod', 'epsilon_i': 1e-8, 'Qian-Grepl': False, 'safety_tolerance': 1e-16, 'beta': 0.95, 'store_estimator': False}
    else:
        if 'radius' not in TR_parameters:
            TR_parameters['radius'] = 0.1
        if 'sub_tolerance' not in TR_parameters:
            TR_parameters['sub_tolerance'] = 1e-8
        if 'max_iterations' not in TR_parameters:
            TR_parameters['max_iterations'] = 40
        if 'max_iterations_subproblem' not in TR_parameters:
            TR_parameters['max_iterations_subproblem'] = 400
        if 'starting_parameter' not in TR_parameters:
            TR_parameters['starting_parameter'] = parameter_space.sample_randomly(1)[0]
        if 'safety_tolerance' not in TR_parameters:
            TR_parameters['safety_tolerance'] = 1e-16
        if 'max_iterations_armijo' not in TR_parameters:
            TR_parameters['max_iterations_armijo'] = 50
        if 'initial_step_armijo' not in TR_parameters:
            TR_parameters['initial_step_armijo'] = 0.5
        if 'armijo_alpha' not in TR_parameters:
            TR_parameters['armijo_alpha'] = 1.e-4
        if 'opt_method' not in TR_parameters:
            TR_parameters['opt_method'] = 'BFGSMethod'
        if 'epsilon_i' not in TR_parameters:
            TR_parameters['epsilon_i'] = 1e-8
        if 'store_estimator' not in TR_parameters:
            TR_parameters['store_estimator'] = False
        if 'Qian-Grepl' not in TR_parameters:
            TR_parameters['Qian-Grepl'] = False
        if 'beta' not in TR_parameters:
            TR_parameters['beta'] = 0.95
        mu_k = TR_parameters['starting_parameter']

        TR_parameters['full_order_model'] = False # tbc

    if extension_params is None:
        extension_params={'Enlarge_radius': True, 'opt_fom': None,
                          'store_subproblem_iterations': True, 'return_data_dict': False}
    elif TR_parameters['Qian-Grepl']:
        extension_params['Enlarge_radius'] = False
        if 'opt_fom' not in extension_params:
            extension_params['opt_fom'] = None
        if 'return_data_dict' not in extension_params:
            extension_params['return_data_dict'] = False
        if 'store_subproblem_iterations' not in extension_params:
            extension_params['store_subproblem_iterations'] = True
    else:
        if 'Enlarge_radius' not in extension_params:
            extension_params['Enlarge_radius'] = True
        if 'opt_fom' not in extension_params:
            extension_params['opt_fom'] = None
        if 'store_subproblem_iterations' not in extension_params:
            extension_params['store_subproblem_iterations'] = True
        if 'return_data_dict' not in extension_params:
            extension_params['return_data_dict'] = False

    if opt_fom is None:
        opt_fom = extension_params['opt_fom']

    if 'FOC_tolerance' not in TR_parameters:
        TR_parameters['FOC_tolerance'] = TR_parameters['sub_tolerance']

    if TR_parameters['Qian-Grepl']:
        print('QIAN et al. 2017 Method')

    if TR_parameters['opt_method'] == 'AdaptiveTaylor_Newton' or TR_parameters['opt_method'] == 'AdaptiveTaylor_BFGS':
        adaptive_taylor = True
    else:
        adaptive_taylor = False

    print('starting parameter {}'.format(mu_k))

    # timings
    tic = time.time()
    Js = []
    FOCs = []
    times = []
    j_list = []
    times_est_evaluations = []
    J_estimator = []
    mu_est = []

    k = 0
    mu_list = []
    mu_list.append(mu_k)
    JFE_list = []
    # the next part is tbc
    if TR_parameters['opt_method']!= 'BFGSMethod': #Keil et al. '20 method does not have this feature
        if 'JFE_start' in TR_parameters:
            JFE_list.append(TR_parameters['JFE_start'])
        else:
            print("******")
            print("!!!!!!!!!!!!!!!!Please, to improve the method, give me the starting value of the cost for FOM model (which you have accessible due to the initialization of the ROM model)!!!!!!!!!!!")
            print("******")
    #----- up to here
    normgrad = 1e6
    estimate_gradient = 1e6 # Used only for Qian et al. method
    old_gradient = 1
    point_rejected = False
    additional_criteria = 0
    model_has_been_enriched = False
    J_k = opt_rom.output_functional_hat(mu_k)
    print("Starting value of the cost: {}".format(J_k))
    print("******************************* \n")
    while k < TR_parameters['max_iterations']:
        if point_rejected:
             point_rejected = False
             if TR_parameters['radius'] < 2.22*1e-16:
                 print('\nTR-radius is below machine precision... stopping')
                 break
        else:
            if not TR_parameters['Qian-Grepl']:
                if normgrad <= TR_parameters['FOC_tolerance'] :
                    print('\nStopping criteria fulfilled: FOC condition: {}'.format(normgrad))
                    additional_criteria = 1
            else:
                if (normgrad + estimate_gradient <= TR_parameters['FOC_tolerance'] ):
                    print('\nStopping criteria fulfilled: normgrad {} + estimate_gradient {}'.format(normgrad,estimate_gradient))
                    additional_criteria = 1
            if 'control_mu' in TR_parameters:
                if TR_parameters['control_mu'] and np.abs(normgrad-old_gradient) <= TR_parameters['safety_tolerance']:
                    print('\n THE METHOD IS STAGNATING, STOP')
                    break
        if 'control_mu' in TR_parameters and additional_criteria:
            if TR_parameters['control_mu']:
                print(' ... additional mu control requested.')
                if opt_fom is None:
                    print(' ... NOT POSSIBLE since opt_fom is not available')
                else:
                    print(' ... estimating with FOM estimator')
                    mu_estimate = opt_fom.estimate_distance_to_true_optimal_parameter_TV(mu_k, parameter_space, U=u, P=p)
                if isinstance(TR_parameters['control_mu'],float):
                    times_est_evaluations.append(times[-1])
                    mu_est.append(mu_estimate)
                    print('mu estimate is {}'.format(mu_estimate), end='')
                    if mu_estimate > TR_parameters['control_mu']:
                        additional_criteria = 0
                        print(' ... but {} was requested ...'.format(
                            TR_parameters['control_mu']))
                        if old_gradient == normgrad and j!=0:
                            print('I CAN NOT FIND A BETTER OPTIMUM... ', end='')
                            additional_criteria = 1
                        else:
                            TR_parameters['FOC_tolerance']=np.maximum(normgrad/100.,5e-9)
                            old_gradient = normgrad
                            print('trying to continue with lower tolerance')
                            print('New FOC tolerance: {}'.format(TR_parameters['FOC_tolerance']))
                    print()
                else:
                    assert 0, 'set a float for parameter "control_mu" '
        if additional_criteria:
            break

        if TR_parameters['opt_method'] == "BFGSMethod":
            mu_kp1, Jcp, j, J_kp1, _  = solve_optimization_subproblem_BFGS(opt_rom, parameter_space, mu_k,
                                                                                 TR_parameters)
        else:
            mu_kp1, Jcp, j, J_kp1, _ = solve_optimization_subproblem_NewtonMethod(opt_rom, parameter_space, mu_k, TR_parameters)
        u_rom = opt_rom.solve(mu_kp1)
        p_rom = opt_rom.solve_dual(mu_kp1, U=u_rom)
        estimate_J = opt_rom.estimate_output_functional_hat(u_rom, p_rom, mu_kp1)
        if TR_parameters['Qian-Grepl']:
            estimate_gradient = opt_rom.estimate_output_functional_hat_gradient_norm(mu_kp1, u_rom, p_rom)

        if J_kp1 + estimate_J < Jcp:
            print('checked sufficient condition, starting enrichment')
            if TR_parameters['opt_method'] == 'AdaptiveTaylor_Newton':
                opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, adaptive_taylor=adaptive_taylor)
            else:
                opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor)
            model_has_been_enriched = True
            JFE_list.append(reductor.fom.output_functional_hat(mu_kp1,u))

            if extension_params['Enlarge_radius']:
                if len(JFE_list) >= 2:
                    if (JFE_list[-2]-JFE_list[-1])/(J_k-J_kp1) > 0.75:
                        if 'JFE_start' not in TR_parameters: # This is need to have method of Keil et al. '20
                            if k-1!= 0:
                                TR_parameters['radius'] *= 2
                                print('enlarging the TR radius to {}'.format(TR_parameters['radius']))
                        else:
                            TR_parameters['radius'] *= 2
                            print('enlarging the TR radius to {}'.format(TR_parameters['radius']))

            print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))
            mu_list.append(mu_kp1)
            times.append(time.time() -tic)
            Js.append(J_kp1)
            J_estimator.append(estimate_J)
            mu_k = mu_kp1
            J_k = opt_rom.output_functional_hat(mu_k)

        elif J_kp1 - estimate_J > Jcp:
            print('necessary condition failed')
            TR_parameters['radius'] = TR_parameters['radius'] * 0.5
            print("Shrinking the TR radius to: {} because Jcp {} and J_kp1 {}".format(TR_parameters['radius'], Jcp,J_kp1))
            point_rejected = True
            if point_rejected and (TR_parameters['opt_method'] =="BFGS_Newton" or TR_parameters['opt_method'] =='AdaptiveTaylor_BFGS_Newton'):
                reductor = starting_reductor
                opt_rom = starting_rom

        else:
            print('enriching to check the sufficient decrease condition')
            if TR_parameters['opt_method'] == 'AdaptiveTaylor_Newton':
                new_rom, new_reductor, u, p = enrichment_step(mu_kp1, reductor, adaptive_taylor=adaptive_taylor)
            else:
                new_rom, new_reductor, u, p = enrichment_step(mu_kp1, reductor)
            model_has_been_enriched = True
            JFE_list.append(reductor.fom.output_functional_hat(mu_kp1, u))


            J_kp1 = new_rom.output_functional_hat(mu_kp1)
            print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))
            if J_kp1 > Jcp + 1e-8:    # add a safety tolerance of 1e-8 for avoiding numerical stability effects
                TR_parameters['radius'] = TR_parameters['radius'] * 0.5
                print(
                    "Shrinking the TR radius to: {} because Jcp {} and J_kp1 {}".format(TR_parameters['radius'],
                                                                                        Jcp,
                                                                                        J_kp1))
                JFE_list.pop(-1) #We need to remove the value from the list, because we reject the parameter
                point_rejected = True

            else:
                opt_rom = new_rom
                reductor = new_reductor
                mu_list.append(mu_kp1)
                times.append(time.time() -tic)
                Js.append(J_kp1)
                J_estimator.append(estimate_J)
                mu_k = mu_kp1
                if extension_params['Enlarge_radius']:
                    if len(JFE_list) >= 2:
                        if (JFE_list[-2] - JFE_list[-1]) / (J_k - J_kp1) > 0.75:
                            if 'JFE_start' not in TR_parameters:  # This is need to have method of Paper_1
                                if k - 1 != 0:
                                    TR_parameters['radius'] *= 2
                                    print('enlarging the TR radius to {}'.format(TR_parameters['radius']))
                            else:
                                TR_parameters['radius'] *= 2
                                print('enlarging the TR radius to {}'.format(TR_parameters['radius']))
                J_k = J_kp1


        if model_has_been_enriched and TR_parameters['Qian-Grepl']:
            # Qian et al. method does not use the fom gradient
            model_has_been_enriched = False

        if not point_rejected:
            if model_has_been_enriched:
                print('computing the fom gradient since the model was enriched')
                gradient = reductor.fom.output_functional_hat_gradient(mu_k, U=u, P=p)
                mu_box = opt_rom.primal_model.parameters.parse(mu_k.to_numpy() - gradient)
                first_order_criticity = mu_k.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
                normgrad = np.linalg.norm(first_order_criticity)
                model_has_been_enriched = False
            else:
                estimate_gradient = opt_rom.estimate_output_functional_hat_gradient_norm(mu_k)
                gradient = opt_rom.output_functional_hat_gradient(mu_k)
                mu_box = opt_rom.primal_model.parameters.parse(mu_k.to_numpy() - gradient)
                first_order_criticity = mu_k.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
                normgrad = np.linalg.norm(first_order_criticity)

            FOCs.append(normgrad)
            j_list.append(j)

            if TR_parameters['Qian-Grepl']:
                print('estimate_gradient {}'.format(estimate_gradient))
            print("First order critical condition: {}".format(normgrad))

            k = k + 1
        print("******************************* \n")


    if extension_params['Enlarge_radius']:
        Js = JFE_list  # This is for speeding-up the computation of the error for the TR method, of course it does not apply to Qian-Grepl
        # This procedure does not give additional speed-up to our method, 
        # but improves only the time of the post-processing step,
        # to have the plot of the error, which is not counted in the computational time of the method.

    if k >= TR_parameters['max_iterations']:
        print (" WARNING: Maximum number of iteration for the TR algorithm reached")

    data = {}
    if extension_params['timings']:
        if TR_parameters['control_mu']:
            data['times_est_evaluations'] = times_est_evaluations
            data['mu_est'] = mu_est
        if return_opt_rom:
            data['opt_rom'] = opt_rom
        if TR_parameters['store_estimator']:
            data['J_estimator'] = J_estimator
        if extension_params['store_subproblem_iterations']:
            data['j_list'] = j_list
        if extension_params['return_data_dict']:
            return mu_list, times, Js, FOCs, data
        else:
            # this is only here to be compatible with the old state
            if return_opt_rom:
                return mu_list, times, Js, FOCs, opt_rom
            else:
                return mu_list, times, Js, FOCs

    return mu_list


def TR_algorithm_with_optional_enrichment(opt_rom, reductor, parameter_space, TR_parameters=None, extension_params=None, opt_fom=None, return_opt_rom=False):
    if TR_parameters is None:
        mu_k = parameter_space.sample_randomly(1)[0]
        TR_parameters = {'radius': 0.1, 'sub_tolerance': 1e-8, 'max_iterations': 40,
                         'max_iterations_subproblem': 400,
                         'starting_parameter': mu_k, 'cost_functional_relative_tolerance': 1e-12,
                         'max_iterations_armijo': 50, 'initial_step_armijo': 0.5, 'armijo_alpha': 1e-4,
                         'opt_method': 'NewtonMethod', 'full_order_model': False, 'epsilon_i': 1e-8}
    else:
        if 'radius' not in TR_parameters:
            TR_parameters['radius'] = 0.1
        if 'sub_tolerance' not in TR_parameters:
            TR_parameters['sub_tolerance'] = 1e-8
        if 'max_iterations' not in TR_parameters:
            TR_parameters['max_iterations'] = 40
        if 'max_iterations_subproblem' not in TR_parameters:
            TR_parameters['max_iterations_subproblem'] = 400
        if 'starting_parameter' not in TR_parameters:
            TR_parameters['starting_parameter'] = parameter_space.sample_randomly(1)[0]
        if 'safety_tolerance' not in TR_parameters:
            TR_parameters['safety_tolerance'] = 1e-16
        if 'cost_functional_relative_tolerance' not in TR_parameters:
            TR_parameters['cost_functional_relative_tolerance'] = TR_parameters['safety_tolerance']
        if 'mu_iterate_relative_tolerance' not in TR_parameters:
            TR_parameters['mu_iterate_relative_tolerance'] = TR_parameters['safety_tolerance']
        if 'max_iterations_armijo' not in TR_parameters:
            TR_parameters['max_iterations_armijo'] = 50
        if 'initial_step_armijo' not in TR_parameters:
            TR_parameters['initial_step_armijo'] = 0.5
        if 'armijo_alpha' not in TR_parameters:
            TR_parameters['armijo_alpha'] = 1.e-4
        if 'opt_method' not in TR_parameters:
            TR_parameters['opt_method'] = 'NewtonMethod'
        if 'full_order_model' not in TR_parameters:
            TR_parameters['full_order_model'] = False
        if 'epsilon_i' not in TR_parameters:
            TR_parameters['epsilon_i'] = 1e-8
        if 'store_estimator' not in TR_parameters:
            TR_parameters['store_estimator'] = False

        mu_k = TR_parameters['starting_parameter']

    if extension_params is None:
        extension_params={'mu_estimators': False,
                          'opt_fom': None,
                          'store_subproblem_iterations': True, 'return_data_dict': False}
    else:
        if 'opt_fom' not in extension_params:
            extension_params['opt_fom'] = None
        if 'mu_estimators' not in extension_params:
            extension_params['mu_estimators'] = False
        if 'store_subproblem_iterations' not in extension_params:
            extension_params['store_subproblem_iterations'] = True
        if 'return_data_dict' not in extension_params:
            extension_params['return_data_dict'] = False

    if opt_fom is None:
        opt_fom = extension_params['opt_fom']

    if reductor.unique_basis:
        TR_parameters["symmetric_Hessian"]= True
    else:
        TR_parameters["symmetric_Hessian"]= False

    if 'FOC_tolerance' not in TR_parameters:
        TR_parameters['FOC_tolerance'] = TR_parameters['sub_tolerance']

    if TR_parameters['opt_method'] == 'AdaptiveTaylor_Newton':
        adaptive_taylor = True
    else:
        adaptive_taylor = False

    print('starting parameter {}'.format(mu_k))

    # timings
    tic = time.time()
    Js = []
    FOCs = []
    times = []
    j_list = []
    times_est_evaluations = []
    J_estimator = []
    mu_est = []

    k = 0
    mu_list = []
    mu_list.append(mu_k)
    JFE_list = []
    if 'JFE_start' in TR_parameters:
        JFE_list.append(TR_parameters['JFE_start'])
    else:
        print("******")
        print("!!!!!!!!!!!!!!!!You forgot to give me the starting value of the cost for FOM model!!!!!!!!!!!")
        print("******")
    normgrad = 1e6
    old_gradient = 1
    point_rejected = False
    additional_criteria = 0
    model_has_been_enriched = False
    Skip_enrichment_step = False
    J_k = opt_rom.output_functional_hat(mu_k)
    print("Starting value of the cost: {}".format(J_k))
    print("******************************* \n")
    while k < TR_parameters['max_iterations']:
        if point_rejected:
             point_rejected = False
             if TR_parameters['radius'] < 2.22*1e-16:
                 print('\nTR-radius is below machine precision... stopping')
                 break
        else:
            if normgrad <= TR_parameters['FOC_tolerance']:
                print('\nStopping criteria fulfilled: FOC condition: {}'.format(normgrad))
                additional_criteria = 1
            if 'control_mu' in TR_parameters:
                if TR_parameters['control_mu'] and np.abs(normgrad-old_gradient) <= TR_parameters['safety_tolerance']:
                    print('\n THE METHOD IS STAGNATING, STOP')
                    print(' ... estimating with FOM estimator')
                    mu_estimate = opt_fom.estimate_distance_to_true_optimal_parameter_TV(mu_k, parameter_space, U=u,
                                                                                         P=p)
                    print('mu estimate is {}'.format(mu_estimate))
                    break

        if 'control_mu' in TR_parameters and additional_criteria:
            if TR_parameters['control_mu']:
                print(' ... additional mu control requested.')
                if opt_fom is None:
                    print(' ... NOT POSSIBLE since opt_fom is not available')
                else:
                    print(' ... estimating with FOM estimator')
                    mu_estimate = opt_fom.estimate_distance_to_true_optimal_parameter_TV(mu_k, parameter_space, U=u, P=p)
                if isinstance(TR_parameters['control_mu'],float):
                    times_est_evaluations.append(times[-1])
                    mu_est.append(mu_estimate)
                    print('mu estimate is {}'.format(mu_estimate), end='')
                    if mu_estimate > TR_parameters['control_mu']:
                        additional_criteria = 0
                        print(' ... but {} was requested ...'.format(
                            TR_parameters['control_mu']))
                        if old_gradient == normgrad and j!=0:
                            print('I CAN NOT FIND A BETTER OPTIMUM... ', end='')
                            print()
                            additional_criteria = 1
                        else:
                            TR_parameters['FOC_tolerance']=np.maximum(normgrad/100.,5e-9)
                            old_gradient = normgrad
                            print('trying to continue with lower tolerance')
                            print('New FOC tolerance: {}'.format(TR_parameters['FOC_tolerance']))
                    print()
                else:
                    assert 0, 'set a float for parameter "control_mu" '
        if additional_criteria:
            break

        mu_kp1, Jcp, j, J_kp1, _ = solve_optimization_subproblem_NewtonMethod(opt_rom, parameter_space, mu_k, TR_parameters)
        u_rom = opt_rom.solve(mu_kp1)
        p_rom = opt_rom.solve_dual(mu_kp1, U=u_rom)
        estimate_J = opt_rom.estimate_output_functional_hat(u_rom, p_rom, mu_kp1)

        if J_kp1 + estimate_J < Jcp:
            print("checked sufficient condition")
            old_gradient = normgrad #Necessary if the parameter control is active
            u = reductor.fom.solve(mu_kp1)
            p = reductor.fom.solve_dual(mu_kp1,U= u)
            gradient = reductor.fom.output_functional_hat_gradient(mu_kp1, U=u, P=p)
            mu_box = opt_rom.primal_model.parameters.parse(mu_kp1.to_numpy() - gradient)
            first_order_criticity = mu_kp1.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
            normgrad = np.linalg.norm(first_order_criticity)
            if normgrad <= TR_parameters['FOC_tolerance'] :
                print('\nStopping criteria fulfilled: FOC condition: {}'.format(normgrad))
                additional_criteria = 1
            if 'control_mu' in TR_parameters and additional_criteria:
                if TR_parameters['control_mu']:
                    print(' ... additional mu control requested.')
                    if opt_fom is None:
                        print(' ... NOT POSSIBLE since opt_fom is not available')
                    else:
                        print(' ... estimating with FOM estimator')
                        mu_estimate = opt_fom.estimate_distance_to_true_optimal_parameter_TV(mu_k, parameter_space, U=u,
                                                                                             P=p)
                    if isinstance(TR_parameters['control_mu'], float):
                        times_est_evaluations.append(times[-1])
                        mu_est.append(mu_estimate)
                        print('mu estimate is {}'.format(mu_estimate), end='')
                        if mu_estimate > TR_parameters['control_mu']:
                            additional_criteria = 0
                            print(' ... but {} was requested ...'.format(
                                TR_parameters['control_mu']))
                            if old_gradient == normgrad and j != 0:
                                print('I CAN NOT FIND A BETTER OPTIMUM... ', end='')
                                print()
                                additional_criteria = 1
                            else:
                                TR_parameters['FOC_tolerance'] = np.maximum(normgrad / 100., 5e-9)
                                print('trying to continue with lower tolerance')
                                print('New FOC tolerance: {}'.format(TR_parameters['FOC_tolerance']))
                        print()
                    else:
                        assert 0, 'set a float for parameter "control_mu" '
            if additional_criteria:
                print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))
                mu_list.append(mu_kp1)
                times.append(time.time() - tic)
                Js.append(J_kp1)
                J_estimator.append(estimate_J)
                FOCs.append(normgrad)
                break

            gradient_rom = opt_rom.output_functional_hat_gradient(mu_kp1, U=u_rom, P=p_rom)
            mu_box = opt_rom.primal_model.parameters.parse(mu_kp1.to_numpy() - gradient_rom)
            first_order_criticity = mu_kp1.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
            normgrad_rom = np.linalg.norm(first_order_criticity)
            JFE_list.append(reductor.fom.output_functional_hat(mu_kp1,u))
            if len(JFE_list) >= 2:
                if 2*TR_parameters['radius'] <= TR_parameters['maximum_radius']:
                    if (JFE_list[-2]-JFE_list[-1])/(J_k-J_kp1) > 0.75:
                        TR_parameters['radius'] *= 2
                        print('enlarging the TR radius to {}'.format(TR_parameters['radius']))

            if estimate_J/J_kp1> TR_parameters['beta_3']*TR_parameters['radius'] or np.abs(normgrad-normgrad_rom)/(normgrad_rom+np.finfo(float).eps) > TR_parameters['tau_g'] or np.linalg.norm(gradient-gradient_rom) / (np.linalg.norm(gradient) + np.finfo(float).eps) > np.minimum(TR_parameters['tau_grad'],TR_parameters['beta_3']*TR_parameters['radius']):
                print('Skip_enrichment_step is false, therefore starting enrichment')
                Skip_enrichment_step = False
                model_has_been_enriched = True
                if TR_parameters['opt_method'] == 'AdaptiveTaylor_Newton':
                    opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, adaptive_taylor=adaptive_taylor, U = u, P = p)
                else:
                    opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, U = u, P = p)
            else:
                print('Skip_enrichment_step is true, no enrichment needed')
                Skip_enrichment_step = True
                u_box = deepcopy(u)
                p_box = deepcopy(p)

            print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))
            mu_list.append(mu_kp1)
            times.append(time.time() -tic)
            Js.append(J_kp1)
            J_estimator.append(estimate_J)
            mu_k = mu_kp1
            if model_has_been_enriched:
                J_k = opt_rom.output_functional_hat(mu_k) # Because the model is updated at mu_k, we need to recompute it
            else:
                J_k = J_kp1

        elif J_kp1 - estimate_J > Jcp:
            print('necessary condition failed')
            TR_parameters['radius'] = TR_parameters['radius'] * 0.5
            print("Shrinking the TR radius to: {} because Jcp {} and J_kp1 {}".format(TR_parameters['radius'], Jcp,J_kp1))
            point_rejected = True
            if Skip_enrichment_step:
                # This is true when in the previous step we skipped the enrichment
                print('In the previous step the enrichment was skipped, so enriching at the previous point')
                if TR_parameters['opt_method'] == 'AdaptiveTaylor_Newton':
                    opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, adaptive_taylor=adaptive_taylor,
                                                              U=u_box, P=p_box)
                else:
                    opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, U=u_box, P=p_box)
            elif TR_parameters['radius'] <= TR_parameters['minimum_radius']:
                print('Radius below minimum radius, enriching to improve the method')
                if TR_parameters['opt_method'] == 'AdaptiveTaylor_Newton':
                    opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, adaptive_taylor=adaptive_taylor)
                else:
                    opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor)
            Skip_enrichment_step = False
        else:
            old_gradient = normgrad #Necessary if the parameter control is active
            u = reductor.fom.solve(mu_kp1)
            p = reductor.fom.solve_dual(mu_kp1, U=u)
            gradient = reductor.fom.output_functional_hat_gradient(mu_kp1, U=u, P=p)
            mu_box = opt_rom.primal_model.parameters.parse(mu_kp1.to_numpy() - gradient)
            first_order_criticity = mu_kp1.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
            normgrad = np.linalg.norm(first_order_criticity)
            if normgrad <= TR_parameters['FOC_tolerance'] :
                print('\nStopping criteria fulfilled: FOC condition: {}'.format(normgrad))
                additional_criteria = 1
            if 'control_mu' in TR_parameters and additional_criteria:
                if TR_parameters['control_mu']:
                    print(' ... additional mu control requested.')
                    if opt_fom is None:
                        print(' ... NOT POSSIBLE since opt_fom is not available')
                    else:
                        print(' ... estimating with FOM estimator')
                        mu_estimate = opt_fom.estimate_distance_to_true_optimal_parameter_TV(mu_k, parameter_space, U=u,
                                                                                             P=p)
                    if isinstance(TR_parameters['control_mu'], float):
                        times_est_evaluations.append(times[-1])
                        mu_est.append(mu_estimate)
                        print('mu estimate is {}'.format(mu_estimate), end='')
                        if mu_estimate > TR_parameters['control_mu']:
                            additional_criteria = 0
                            print(' ... but {} was requested ...'.format(
                                TR_parameters['control_mu']))
                            if old_gradient == normgrad and j != 0:
                                print('I CAN NOT FIND A BETTER OPTIMUM... ', end='')
                                print()
                                additional_criteria = 1
                            else:
                                TR_parameters['FOC_tolerance'] = np.maximum(normgrad / 100., 5e-9)
                                print('trying to continue with lower tolerance')
                                print('New FOC tolerance: {}'.format(TR_parameters['FOC_tolerance']))
                        print()
                    else:
                        assert 0, 'set a float for parameter "control_mu" '
            if additional_criteria:
                print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))
                mu_list.append(mu_kp1)
                times.append(time.time() - tic)
                Js.append(J_kp1)
                J_estimator.append(estimate_J)
                FOCs.append(normgrad)
                break
            gradient_rom = opt_rom.output_functional_hat_gradient(mu_kp1, U=u_rom, P=p_rom)
            mu_box = opt_rom.primal_model.parameters.parse(mu_kp1.to_numpy() - gradient_rom)
            first_order_criticity = mu_kp1.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
            normgrad_rom = np.linalg.norm(first_order_criticity)
            JFE_list.append(reductor.fom.output_functional_hat(mu_kp1, u))
            if 2*TR_parameters['radius'] <= TR_parameters['maximum_radius'] and estimate_J / J_kp1 <= 2 * TR_parameters['radius'] * TR_parameters['beta_3'] and np.abs(normgrad_rom-normgrad)/(normgrad_rom + np.finfo(float).eps) <= TR_parameters['tau_g'] and np.linalg.norm(gradient-gradient_rom) / (np.linalg.norm(gradient) + np.finfo(float).eps) <= np.minimum(TR_parameters['tau_grad'], 2* TR_parameters['radius'] * TR_parameters['beta_3'])  and len(JFE_list) >= 2 and (JFE_list[-2] - JFE_list[-1]) / (J_k - J_kp1) > 0.75:
                print('Skip_enrichment_step is true, no enrichment needed' )
                Skip_enrichment_step = True
                u_box = deepcopy(u)
                p_box = deepcopy(p)
                mu_list.append(mu_kp1)
                times.append(time.time() - tic)
                Js.append(J_kp1)
                J_estimator.append(estimate_J)
                mu_k = mu_kp1
                TR_parameters['radius'] *= 2
                print('enlarging the TR radius to {}'.format(TR_parameters['radius']))
                J_k = J_kp1  # we do not update the model so J_kp1 is fine as it is

                print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))
            else:
                J_kp1 = JFE_list[-1]
                if J_kp1 > Jcp + 1e-8:    # add a safety tolerance of 1e-8 for avoiding numerical stability effects
                    print("Sufficient decrease condition NOT fulfilled because Jcp {} and J_kp1 {}".format(Jcp,J_kp1))
                    TR_parameters['radius'] = TR_parameters['radius'] * 0.5
                    print("Shrinking the TR radius to: {} ".format(TR_parameters['radius']))
                    JFE_list.pop(-1) #We need to remove the value from the list, because we reject the parameter
                    point_rejected = True
                    if Skip_enrichment_step:
                        # This is true when in the previous step we skipped the enrichment
                        print('In the previous step the enrichment was skipped, so enriching at the previous point')
                        if TR_parameters['opt_method'] == 'AdaptiveTaylor_Newton':
                            opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, adaptive_taylor=adaptive_taylor,
                                                                      U=u_box, P=p_box)
                        else:
                            opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, U=u_box, P=p_box)
                    elif TR_parameters['radius'] <= TR_parameters['minimum_radius']:
                        print('Radius below minimum radius, enriching to improve the method')
                        if TR_parameters['opt_method'] == 'AdaptiveTaylor_Newton':
                            opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, adaptive_taylor=adaptive_taylor,
                                                                      U=u, P=p)
                        else:
                            opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, U=u, P=p)
                    Skip_enrichment_step = False
                else:
                    Skip_enrichment_step = False
                    print("Sufficient decrease condition fulfilled")
                    print('Enriching the RB model')
                    if TR_parameters['opt_method'] == 'AdaptiveTaylor_Newton':
                        opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor,
                                                                      adaptive_taylor=adaptive_taylor, U=u, P=p)
                    else:
                        opt_rom, reductor, u, p = enrichment_step(mu_kp1, reductor, U=u, P=p)

                    mu_list.append(mu_kp1)
                    times.append(time.time() -tic)
                    Js.append(J_kp1)
                    J_estimator.append(estimate_J)
                    mu_k = mu_kp1

                    if len(JFE_list) >= 2:
                        if 2 * TR_parameters['radius'] <= TR_parameters['maximum_radius']:
                            if (JFE_list[-2]-JFE_list[-1])/(J_k-J_kp1) > 0.75:
                                TR_parameters['radius'] *= 2
                                print('enlarging the TR radius to {}'.format(TR_parameters['radius']))
                    J_k = J_kp1 # J_kp1 is already computed for the updated model

                    print("k: {} - j {} - Cost Functional: {} - mu: {}".format(k, j, J_kp1, mu_kp1))

        if not point_rejected:

            FOCs.append(normgrad)
            j_list.append(j)

            print("First order critical condition: {}".format(normgrad))
            if 'mu_estimators' in extension_params:
                if extension_params['mu_estimators']:
                    if opt_fom is not None:
                        print("estimator for mu with FOM: {}".format(opt_fom.estimate_distance_to_true_optimal_parameter_TV(mu_k, parameter_space, U=u, P=p)))

            k = k + 1
        print("******************************* \n")


    if extension_params['Enlarge_radius']:
        # This is for speeding-up the computation of the error for the TR method, of course it does not apply to Qian-Grepl
        # This procedure does not give additional speed-up to our method,
        # but improves only the time of the post-processing step,
        # to have the plot of the error, which is not counted in the computational time of the method.
        if len(Js) == len(JFE_list):
            # the last entry in JFE_list is missing. pick the last entry from Js and add another for detection
            Js_copy = Js.copy()
            Js = JFE_list
            Js.append(Js_copy[-1])
            Js.append(0)
        else:
            Js = JFE_list


    if k >= TR_parameters['max_iterations']:
        print (" WARNING: Maximum number of iteration for the TR algorithm reached")

    data = {}
    if extension_params['timings']:
        if TR_parameters['control_mu']:
            data['times_est_evaluations'] = times_est_evaluations
            data['mu_est'] = mu_est
        if return_opt_rom:
            data['opt_rom'] = opt_rom
        if TR_parameters['store_estimator']:
            data['J_estimator'] = J_estimator
        if extension_params['store_subproblem_iterations']:
            data['j_list'] = j_list
        if extension_params['return_data_dict']:
            return mu_list, times, Js, FOCs, data
        else:
            # this is only here to be compatible with the old state
            if return_opt_rom:
                return mu_list, times, Js, FOCs, opt_rom
            else:
                return mu_list, times, Js, FOCs

    return mu_list

def search_Cauchy_Steihaug(opt_model,parameter_space,x_nocedal,mu,direction,gradient, TR_radius):
    # This function follows Section 16.7 of Nocedal and Wright 2006
    ranges = parameter_space.ranges
    dir_dict =  opt_model.primal_model.parameters.parse(direction)
    t_list = []

    for (key,size) in sorted(parameter_space.parameters.items()):
        range_ = ranges[key]
        for j in range(size):

            if dir_dict[key][j]>0:
                t_list.append((np.minimum(TR_radius,range_[1]-mu[key][j]-x_nocedal[key][j]))/dir_dict[key][j])
            elif dir_dict[key][j]<0:
                t_list.append((np.maximum(-TR_radius,range_[0]-mu[key][j]-x_nocedal[key][j]))/dir_dict[key][j])
            else:
                t_list.append(1e16)

    p_j = deepcopy(direction)
    x_t_jm1 = x_nocedal.to_numpy()

    t_array = np.array(t_list)
    t_array_breakpoints = np.unique(t_array)
    t_array_breakpoints = np.append(0.0, t_array_breakpoints)

    for j in range(1, t_array_breakpoints.size):
        t_jm1 = t_array_breakpoints[j - 1]
        t_j = t_array_breakpoints[j]
        if np.linalg.norm(p_j) != 0.0:
            Hpj = opt_model.output_functional_hessian_operator(mu, p_j)
            f_jm1_prime = gradient.dot(p_j) + x_t_jm1.dot(Hpj)
            if f_jm1_prime > 0:
                return x_t_jm1
            else:
                f_jm1_second = p_j.dot(Hpj)
                dtstar = -f_jm1_prime / f_jm1_second
                if dtstar >= 0 and dtstar < t_j - t_jm1:
                    xstar = x_t_jm1 + (dtstar) * p_j
                    return xstar
            x_t_jm1 = x_t_jm1 + (t_j - t_jm1) * p_j
            idx = np.where(t_array <= t_j)
            p_j[idx] = 0.0

    print(
        "The model function admits a point that has all components on the boundary \n Returning the direction to reach it to continue")
    return x_t_jm1

def TR_Steihaug(opt_model, parameter_space, TR_parameters=None):
    if TR_parameters is None:
        mu_k = parameter_space.sample_randomly(1)[0]
        TR_parameters = {'radius': 0.1, 'max_iterations': 400,
                         'starting_parameter': mu_k,
                          'epsilon_i': 1e-8, 'FOC_tolerance': 1e-6,
                         'max_radius': 100, 'control_mu': none}
    else:
        if 'radius' not in TR_parameters:
            TR_parameters['radius'] = 0.1
        if 'max_iterations' not in TR_parameters:
            TR_parameters['max_iterations'] = 400
        if 'starting_parameter' not in TR_parameters:
            TR_parameters['starting_parameter'] = parameter_space.sample_randomly(1)[0]
        if 'epsilon_i' not in TR_parameters:
            TR_parameters['epsilon_i'] = 1e-8
        if 'FOC_tolerance' not in TR_parameters:
            TR_parameters['FOC_tolerance']= 1e-6
        if 'max_radius' not in TR_parameters:
            TR_parameters['max_radius'] = 100
        if 'control_mu' not in TR_parameters:
            TR_parameters['control_mu'] = None

        mu_k = TR_parameters['starting_parameter']

        tic = time.time()
        mu_list = []
        mu_list.append(mu_k)
        FOCs = []
        Js = []
        times = []
        times_est_evaluations = []
        mu_est = []
        old_gradient = 0

        gradient = opt_model.output_functional_hat_gradient(mu_k)
        mu_box = opt_model.primal_model.parameters.parse(mu_k.to_numpy() - gradient)
        first_order_criticity = mu_k.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
        normgrad = np.linalg.norm(first_order_criticity)
        FOCs.append(normgrad)
        J_k = opt_model.output_functional_hat(mu_k)
        Js.append(J_k)
        epsilon_i = TR_parameters['epsilon_i']

        additional_criteria = 0
        k = 0
        while k<= TR_parameters['max_iterations']:
            if TR_parameters['radius'] < 2.22*1e-16:
                print('\nTR-radius is below machine precision... stopping')
                break
            if normgrad <= TR_parameters['FOC_tolerance'] :
                print('\nStopping criteria fulfilled: FOC condition: {}'.format(normgrad))
                additional_criteria= 1
            if 'control_mu' in TR_parameters and additional_criteria:
                if TR_parameters['control_mu']:
                    print(' ... additional mu control requested.')
                    print(' ... estimating with FOM estimator')
                    mu_estimate = opt_model.estimate_distance_to_true_optimal_parameter_TV(mu_k, parameter_space)
                    if isinstance(TR_parameters['control_mu'], float):
                        times_est_evaluations.append(times[-1])
                        mu_est.append(mu_estimate)
                        print('mu estimate is {}'.format(mu_estimate), end='')
                        if mu_estimate > TR_parameters['control_mu']:
                            additional_criteria = 0
                            print(' ... but {} was requested ...'.format(
                                TR_parameters['control_mu']))
                            if old_gradient == normgrad:
                                print('I CAN NOT FIND A BETTER OPTIMUM... ', end='')
                                print()
                                additional_criteria = 1
                            else:
                                TR_parameters['FOC_tolerance'] = np.maximum(normgrad / 100., 5e-9)
                                old_gradient = normgrad
                                print('trying to continue with lower tolerance')
                                print('New FOC tolerance: {}'.format(TR_parameters['FOC_tolerance']))
                        print()
                    else:
                        assert 0, 'set a float for parameter "control_mu" '
            if additional_criteria:
                break

            # We compute the Cauchy point doing one step of projected gradient
            print("Computing at first the Cauchy point")
            delta_mu_k_c = search_Cauchy_Steihaug(opt_model,parameter_space,opt_model.primal_model.parameters.parse(0.0*mu_k.to_numpy()),mu_k,-gradient,gradient,TR_parameters['radius'])

            mu_k_c = opt_model.primal_model.parameters.parse(mu_k.to_numpy()+delta_mu_k_c)

            Active_i, Inactive_i = active_and_inactive_sets(parameter_space, mu_k_c, epsilon_i)

            mu_k_c_not_used = True
            if Inactive_i.sum() == 0.0:
                mu_k_next = mu_k_c
                mu_k_c_not_used = False
            else:
                print("Using CG-Steihaug for the linear system")
                deltamu, itcg, rescg, infocg = TruncCGSteihaug(
                    A_func=lambda v: modified_hessian_action(mu=mu_k, Active=Active_i, Inactive=Inactive_i,
                                                             opt_model=opt_model, eta=v), b= -np.multiply(gradient,Inactive_i), x_0 = delta_mu_k_c,
                    TR_radius=TR_parameters['radius'], tol=1.e-10)


                if infocg > 0:
                        mu_k_next = mu_k_c
                        mu_k_c_not_used = False
                elif np.dot(deltamu, gradient) >= -1.e-14:
                    print('Not a descendent direction ... taking gradient as direction')
                    mu_k_next = mu_k_c
                    mu_k_c_not_used = False
                else:
                    deltamu = search_Cauchy_Steihaug(opt_model, parameter_space, opt_model.primal_model.parameters.parse(delta_mu_k_c),
                                                     mu_k, deltamu, gradient,TR_parameters['radius'])
                    mu_k_next = opt_model.primal_model.parameters.parse(mu_k.to_numpy()+deltamu)
            J_k_next = opt_model.output_functional_hat(mu_k_next)
            proj_dir = mu_k_next.to_numpy()-mu_k.to_numpy()
            den = -gradient.dot(proj_dir)-0.5*proj_dir.dot(opt_model.output_functional_hessian_operator(mu_k, proj_dir))
            rho_k = (J_k-J_k_next)/den

            if rho_k<= 0.25:
                if mu_k_c_not_used:
                    mu_k_next = mu_k_c
                    J_k_next = opt_model.output_functional_hat(mu_k_next)
                    proj_dir = mu_k_next.to_numpy() - mu_k.to_numpy()
                    den = -gradient.dot(proj_dir) - 0.5 * proj_dir.dot(opt_model.output_functional_hessian_operator(mu_k, proj_dir))
                    rho_k = (J_k - J_k_next) / den
            if rho_k<= 0.25:
                TR_parameters['radius'] *= 0.5
                print("Shrinking the radius to {}".format(TR_parameters['radius']))
                if np.max(mu_k_next.to_numpy()-mu_k.to_numpy()) <= 1e-12 or np.abs(J_k_next-J_k)<= 1e-10: 
                    print("Method is stagnating, safety tolerance reached so stopping...")
                    break
            elif rho_k >= 0.75:
                if TR_parameters['radius'] < TR_parameters['max_radius']:
                    TR_parameters['radius'] *= 2
                    print("Enlarging the radius to {}".format(TR_parameters['radius']))
                mu_k = mu_k_next
                mu_list.append(mu_k)
                gradient = opt_model.output_functional_hat_gradient(mu_k)
                mu_box = opt_model.primal_model.parameters.parse(mu_k.to_numpy() - gradient)
                first_order_criticity = mu_k.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
                normgrad = np.linalg.norm(first_order_criticity)
                FOCs.append(normgrad)
                J_k = J_k_next
                Js.append(J_k)
                times.append(time.time()-tic)
            else:
                mu_k = mu_k_next
                mu_list.append(mu_k)
                gradient = opt_model.output_functional_hat_gradient(mu_k)
                mu_box = opt_model.primal_model.parameters.parse(mu_k.to_numpy() - gradient)
                first_order_criticity = mu_k.to_numpy() - projection_onto_range(parameter_space, mu_box).to_numpy()
                normgrad = np.linalg.norm(first_order_criticity)
                FOCs.append(normgrad)
                J_k = J_k_next
                Js.append(J_k)
                times.append(time.time()-tic)

            k = k+1

            print("k: {} - Cost Functional: {} - mu: {}".format(k, J_k, mu_k))
            print("First order critical condition: {}".format(normgrad))
            print("***********************************************\n")

        return mu_list, times, Js, FOCs



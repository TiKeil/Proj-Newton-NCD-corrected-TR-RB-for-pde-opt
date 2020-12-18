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
from numbers import Number
from copy import deepcopy


from pymor.basic import ConstantFunction
from pymor.core.base import BasicObject, ImmutableObject
from pymor.models.basic import StationaryModel
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.discretizers.builtin.cg import InterpolationOperator, L2ProductP1
from pymor.operators.constructions import VectorOperator, LincombOperator, ConstantOperator, VectorFunctional, ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import ProjectionBasedReductor
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor
from pymor.reductors.basic import StationaryRBReductor
from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional
from pymor.parameters.functionals import BaseMaxThetaParameterFunctional
from pymor.parameters.functionals import MaxThetaParameterFunctional
from pymor.parameters.base import Mu
from pymor.discretizers.builtin.grids.boundaryinfos import EmptyBoundaryInfo


class QuadraticPdeoptStationaryCoerciveReductor(CoerciveRBReductor):
    def __init__(self, fom, RBPrimal=None, RBDual=None,
                 opt_product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None, unique_basis=False,
                 reductor_type='simple_coercive', mu_bar=None,
                 prepare_for_hessian=False,
                 prepare_for_gradient_estimate=False, adjoint_estimate=False):

        self.__auto_init(locals())
        if self.opt_product is None:
            self.opt_product = fom.opt_product
        super().__init__(fom, RBPrimal, product=opt_product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol, coercivity_estimator=coercivity_estimator)

        self.non_assembled_primal_reductor = None
        self.non_assembled_primal_rom = None
        self.non_assembled_dual_rom = None

        self.adjoint_approach = self.fom.adjoint_approach
        self.separated_bases = False if self.unique_basis else True

        if unique_basis is True:
            self._build_unique_basis()
            self.bases = {'RB' : self.RBPrimal}
            print('Starting with only one basis with length {}'.format(len(self.RBPrimal)))
        else:
            self.bases = {'RB' : RBPrimal, 'DU': RBDual}
            print('Starting with two bases. ', end='')
            print('Primal and dual have length {} and {}'.format(
                len(RBPrimal), len(RBDual))) if RBPrimal is not None and RBDual is not None else print('The Primal and/or the dual bases are empty')

        # primal model
        self.primal_fom = self.fom.primal_model
        self.primal_rom, self.primal_reductor = self._build_primal_rom()
        self.primal = self.primal_reductor

        # dual model
        if self.RBPrimal is not None:
            self.dual_intermediate_fom, self.dual_rom, self.dual_reductor = self._build_dual_models()
            self.dual = self.dual_reductor

        # pre compute constants for estimators
        k_form = self.fom.output_functional_dict['bilinear_part']
        if isinstance(k_form, LincombOperator):
            alpha_mu_bar = self.fom.compute_continuity_bilinear(k_form, self.fom.opt_product, mu_bar)
            self.cont_k = MaxThetaParameterFunctional(k_form.coefficients, mu_bar, gamma_mu_bar=alpha_mu_bar)
        else:
            self.cont_k = lambda mu: self.fom.compute_continuity_bilinear(k_form, self.fom.opt_product)

        j_form = self.fom.output_functional_dict['linear_part']
        if isinstance(j_form, LincombOperator):
            conts_j = []
            for op in j_form.operators:
                conts_j.append(self.fom.compute_continuity_linear(op, self.fom.opt_product))
            self.cont_j = lambda mu: np.dot(conts_j,np.abs(j_form.evaluate_coefficients(mu)))
        else:
            self.cont_j = lambda mu: self.fom.compute_continuity_linear(j_form, self.fom.opt_product)

        if self.coercivity_estimator is None:
            print('WARNING: coercivity_estimator is None ... setting it to constant 1.')
            self.coercivity_estimator = lambda mu: 1.

        # precompute ||d_mui l_h || 
        if self.prepare_for_gradient_estimate:
            self.cont_l_dmu = self._construct_zero_dict(self.primal_fom.parameters)
            for (key, size) in sorted(self.primal_fom.parameters.items()):
                for l in range(size):
                    conts_l = []
                    l_dmu = self.fom.primal_model.rhs.d_mu(key, l)
                    for op in l_dmu.operators:
                        conts_l.append(self.fom.compute_continuity_linear(op, self.fom.opt_product))
                        self.cont_l_dmu[key][l] = lambda mu: np.dot(conts_l,np.abs(l_dmu.evaluate_coefficients(mu)))

        # precomput parts of || d_mui a_h ||
        if self.prepare_for_gradient_estimate:
            self.cont_a_dmu_functional = self._construct_zero_dict(self.primal_fom.parameters)
            for (key, size) in sorted(self.primal_fom.parameters.items()):
                for l in range(size):
                    self.cont_a_dmu_functional[key][l] = BaseMaxThetaParameterFunctional( \
                                                                 self.fom.primal_model.operator.d_mu(key, l).coefficients,
                                                                 self.fom.primal_model.operator.coefficients, mu_bar)

        self.cont_a = MaxThetaParameterFunctional(self.primal_fom.operator.coefficients, mu_bar)

        # construct e_l from key and item
        self.from_key_item_to_e_l = {}
        eta = self.primal_fom.parameters.parse(np.zeros(self.primal_fom.parameters.dim))
        for (key, size) in sorted(self.primal_fom.parameters.items()):
            key_to_e_l = np.empty(size, dtype=object)
            for index in range(size):
                key_in_eta = eta[key].copy()
                key_in_eta[index] = 1
                if key == 'heaters':
                    e_l = eta.with_(heaters=key_in_eta).to_numpy()
                elif key == 'walls':
                    e_l = eta.with_(walls=key_in_eta).to_numpy()
                elif key == 'doors':
                    e_l = eta.with_(doors=key_in_eta).to_numpy()
                elif key == 'windows':
                    e_l = eta.with_(heaters=key_in_eta).to_numpy()
                key_to_e_l[index] = e_l
            self.from_key_item_to_e_l[key] = key_to_e_l

    def reduce(self):
        assert self.RBPrimal is not None, 'I can not reduce without a RB basis'
        return super().reduce()

    def build_rom(self, projected_operators, estimator):
        if (not self.fom.adjoint_approach or not self.separated_bases) and self.prepare_for_hessian:
            projected_hessian = self.project_hessian()
        if self.fom.adjoint_approach and self.prepare_for_hessian and self.separated_bases:
            projected_hessian = self.project_adjoint_hessian()
        else:
            projected_hessian = None
        projected_product = self.project_product()
        return self.fom.with_(primal_model=self.primal_rom, dual_model=self.dual_rom,
                              opt_product=projected_product,
                              estimators=estimator, output_functional_dict=self.projected_output,
                              projected_hessian=projected_hessian,
                              separated_bases=self.separated_bases)

    def extend_bases(self, mu, printing=True, U = None, P = None):
        if self.unique_basis:
            U, P = self.extend_unique_basis(mu, U, P)
            return U, P

        if U is None:
            U = self.fom.solve(mu)
        if P is None:
            P = self.fom.solve_dual(mu,U=U)
        try:
            self.primal_reductor.extend_basis(U)
            # self.non_assembled_primal_reductor.extend_basis(U)
        except:
            pass
        self.primal_rom = self.primal_reductor.reduce()
        if self.non_assembled_primal_rom is not None:
            self.non_assembled_primal_rom = self.non_assembled_primal_reductor.reduce()
        self.bases['RB'] = self.primal_reductor.bases['RB']
        self.RBPrimal = self.bases['RB']
        self.RBDual.append(P)
        self.RBDual = gram_schmidt(self.RBDual, product=self.opt_product)
        an, bn = len(self.RBPrimal), len(self.RBDual)
        self.dual_intermediate_fom, self.dual_rom, self.dual_reductor = self._build_dual_models()
        self.dual = self.dual_reductor
        self.bases['DU'] = self.dual_reductor.bases['RB']

        if printing:
            print('Enrichment completed... length of Bases are {} and {}'.format(an,bn))
        return U, P

    def extend_unique_basis(self,mu, U = None, P = None):
        assert self.unique_basis
        if U is None:
            U = self.fom.solve(mu=mu)
        if P is None:
            P = self.fom.solve_dual(mu=mu, U=U)
        try:
            self.primal_reductor.extend_basis(U)
        except:
            pass
        try:
            self.primal_reductor.extend_basis(P)
        except:
            pass

        self.primal_rom = self.primal_reductor.reduce()
        if self.non_assembled_primal_rom is not None:
            self.non_assembled_primal_rom = self.non_assembled_primal_reductor.reduce()
        self.bases['RB'] = self.primal_reductor.bases['RB']
        self.RBPrimal = self.bases['RB']

        self.RBDual = self.RBPrimal
        self.dual_intermediate_fom, self.dual_rom, self.dual_reductor = self._build_dual_models()
        self.dual = self.primal_reductor

        an = len(self.RBPrimal)
        print('Length of Basis is {}'.format(an))
        return U, P

    def extend_adaptive_taylor(self, mu, U = None, P = None):
        if U is None:
            U = self.fom.solve(mu)
        if P is None:
            P = self.fom.solve_dual(mu, U=U)
        eta = self.fom.output_functional_hat_gradient(mu, U=U, P=P)
        if self.unique_basis:
            try:
                self.primal_reductor.extend_basis(U)
                self.primal_reductor.extend_basis(P)
            except:
                pass
            u_d_eta = self.fom.solve_for_u_d_eta(mu, eta=eta, U=U)
            p_d_eta = self.fom.solve_for_p_d_eta(mu, eta=eta, U=U, P=P, u_d_eta=u_d_eta)
            try:
                self.primal_reductor.extend_basis(u_d_eta)
                self.primal_reductor.extend_basis(p_d_eta)
            except:
                pass
            self.RBDual = self.primal_reductor.bases['RB']
        else:
            self.RBDual.append(P)
            try:
                self.primal_reductor.extend_basis(U)
            except:
                pass
            u_d_eta = self.fom.solve_for_u_d_eta(mu, eta=eta, U=U)
            p_d_eta = self.fom.solve_for_p_d_eta(mu, eta=eta, U=U, P=P, u_d_eta=u_d_eta)
            try:
                self.primal_reductor.extend_basis(u_d_eta)
            except:
                pass
            self.RBDual.append(p_d_eta)
            self.RBDual = gram_schmidt(self.RBDual, product=self.opt_product)

        self.primal_rom = self.primal_reductor.reduce()
        if self.non_assembled_primal_rom is not None:
            self.non_assembled_primal_rom = self.non_assembled_primal_reductor.reduce()
        self.bases['RB'] = self.primal_reductor.bases['RB']
        self.RBPrimal = self.bases['RB']

        self.dual_intermediate_fom, self.dual_rom, self.dual_reductor = self._build_dual_models()
        self.dual = self.dual_reductor
        self.bases['DU'] = self.dual_reductor.bases['RB']

        an, bn = len(self.RBPrimal), len(self.RBDual)
        if self.unique_basis:
            print('Length of Bases is {}'.format(an,bn))
        else:
            print('Length of Bases are {} and {}'.format(an,bn))
        return U, P

    def _build_unique_basis(self):
        self.RBPrimal.append(self.RBDual)
        self.RBPrimal = gram_schmidt(self.RBPrimal, product=self.opt_product)
        self.RBDual = self.RBPrimal

    def _build_primal_rom(self):
        if self.reductor_type == 'simple_coercive':
            print('building simple coercive primal reductor...')
            primal_reductor = SimpleCoerciveRBReductor(self.fom.primal_model, RB=self.RBPrimal, product=self.opt_product,
                                            coercivity_estimator=self.coercivity_estimator)
        elif self.reductor_type == 'non_assembled':
            print('building non assembled for primal reductor...')
            primal_reductor = NonAssembledCoerciveRBReductor(self.fom.primal_model, RB=self.RBPrimal, product=self.opt_product,
                                                        coercivity_estimator=self.coercivity_estimator)
        else:
            print('building coercive primal reductor...')
            primal_reductor = CoerciveRBReductor(self.fom.primal_model, RB=self.RBPrimal, product=self.opt_product,
                                            coercivity_estimator=self.coercivity_estimator)

        primal_rom = primal_reductor.reduce()
        return primal_rom, primal_reductor

    def _build_dual_models(self):
        assert self.primal_rom is not None
        assert self.RBPrimal is not None
        RBbasis = self.RBPrimal
        rhs_operators = list(self.fom.output_functional_dict['d_u_linear_part'].operators)
        rhs_coefficients = list(self.fom.output_functional_dict['d_u_linear_part'].coefficients)

        bilinear_part = self.fom.output_functional_dict['d_u_bilinear_part']

        for i in range(len(RBbasis)):
            u = RBbasis[i]
            if isinstance(bilinear_part, LincombOperator):
                for j, op in enumerate(bilinear_part.operators):
                    rhs_operators.append(VectorOperator(op.apply(u)))
                    rhs_coefficients.append(ExpressionParameterFunctional('basis_coefficients[{}]'.format(i),
                                                                  {'basis_coefficients': len(RBbasis)})
                                        * bilinear_part.coefficients[j])
            else:
                rhs_operators.append(VectorOperator(bilinear_part.apply(u, None)))
                rhs_coefficients.append(1. * ExpressionParameterFunctional('basis_coefficients[{}]'.format(i),
                                                                  {'basis_coefficients': len(RBbasis)}))

        dual_rhs_operator = LincombOperator(rhs_operators,rhs_coefficients)

        dual_intermediate_fom = self.fom.primal_model.with_(rhs = dual_rhs_operator)

        if self.reductor_type == 'simple_coercive':
            print('building simple coercive dual reductor...')
            dual_reductor = SimpleCoerciveRBReductor(dual_intermediate_fom, RB=self.RBDual,
                                           product=self.opt_product,
                                           coercivity_estimator=self.coercivity_estimator)
        elif self.reductor_type == 'non_assembled':
            print('building non assembled dual reductor...')
            dual_reductor = NonAssembledCoerciveRBReductor(dual_intermediate_fom, RB=self.RBDual,
                                            product=self.opt_product, coercivity_estimator=self.coercivity_estimator)
        else:
            print('building coercive dual reductor...')
            dual_reductor = CoerciveRBReductor(dual_intermediate_fom, RB=self.RBDual,
                                          product=self.opt_product,
                                          coercivity_estimator=self.coercivity_estimator)


        dual_rom = dual_reductor.reduce()
        return dual_intermediate_fom, dual_rom, dual_reductor

    def _construct_zero_dict(self, parameters):
        #prepare dict
        zero_dict = {}
        for key, size in parameters.items():
            zero_ = np.empty(size, dtype=object)
            zero_dict[key] = zero_
        return zero_dict

    #prepare dict
    def _construct_zero_dict_dict(self, parameters):
        zero_dict = {}
        for key, size in parameters.items():
            zero_ = np.empty(size, dtype=dict)
            zero_dict[key] = zero_
            for l in range(size):
                zero_dict[key][l] = self._construct_zero_dict(parameters)
        return zero_dict

    def assemble_error_estimator(self):
        self.projected_output = self.project_output()

        # print_pieces 
        print_pieces = 0

        estimators = {}

        # primal
        class PrimalCoerciveRBEstimator(ImmutableObject):
            def __init__(self, primal_rom, non_assembled_rom=None):
                self.__auto_init(locals())
            def estimate_error(self, U, mu, non_assembled=False):
                if non_assembled and self.non_assembled_rom is not None:
                    return self.non_assembled_rom.estimate_error(U, mu)
                else:
                    return self.primal_rom.estimate_error(U, mu)

        estimators['primal'] = PrimalCoerciveRBEstimator(self.primal_rom, self.non_assembled_primal_rom)

        ##########################################

        # dual
        class DualCoerciveRBEstimator(ImmutableObject):
            def __init__(self, coercivity_estimator, cont_k, primal_estimator, dual_rom, non_assembled_rom=None):
                self.__auto_init(locals())

            def estimate_error(self, U, P, mu, non_assembled=False):
                primal_estimate = self.primal_estimator.estimate_error(U, mu, non_assembled=non_assembled)[0]
                if non_assembled and self.non_assembled_rom is not None:
                    dual_intermediate_estimate = self.non_assembled_rom.estimate_error(P, mu)[0]
                else:
                    dual_intermediate_estimate = self.dual_rom.estimate_error(P, mu)
                if print_pieces or 0:
                    print(self.cont_k(mu), self.coercivity_estimator(mu), primal_estimate, dual_intermediate_estimate)
                return 2* self.cont_k(mu) /self.coercivity_estimator(mu) * primal_estimate + dual_intermediate_estimate

        estimators['dual'] = DualCoerciveRBEstimator(self.coercivity_estimator, self.cont_k, estimators['primal'], self.dual_rom, self.non_assembled_dual_rom)
        ##########################################

        # output hat
        class output_hat_RBEstimator(ImmutableObject):
            def __init__(self, coercivity_estimator, cont_k, cont_j, primal_estimator, dual_estimator,
                         projected_output, dual_rom, P_product, U_product, corrected_output):
                self.__auto_init(locals())

            def estimate_error(self, U, P, mu):
                primal_estimate = self.primal_estimator.estimate_error(U, mu)[0]
                dual_estimate = self.dual_estimator.estimate_error(U, P, mu)

                dual_intermediate_estimate = self.dual_rom.estimate_error(P, mu)[0]

                residual_lhs = self.projected_output['primal_dual_projected_op'].apply2(U, P, mu=mu)[0,0]
                residual_rhs = self.projected_output['dual_projected_rhs'].apply_adjoint(P, mu=mu).to_numpy()[0,0]

                if print_pieces or 0:
                    print(self.coercivity_estimator(mu), primal_estimate, dual_estimate, primal_estimate**2,
                          self.cont_k(mu), primal_estimate, self.coercivity_estimator(mu), norm_P)

                if self.corrected_output:
                    est1 = self.coercivity_estimator(mu) * primal_estimate * dual_estimate + \
                       primal_estimate**2 * self.cont_k(mu)
                    return est1
                else:
                    est2 = self.coercivity_estimator(mu) * primal_estimate * dual_estimate + \
                       primal_estimate**2 * self.cont_k(mu) + \
                       + np.abs(residual_rhs - residual_lhs)
                    return est2

        estimators['output_functional_hat'] = output_hat_RBEstimator(self.coercivity_estimator,
                                                                     self.cont_k, self.cont_j,
                                                                     estimators['primal'], estimators['dual'],
                                                                     self.projected_output, self.dual_rom, self.dual_rom.opt_product,
                                                                     self.primal_rom.opt_product,
                                                                     self.fom.use_corrected_functional)


        ##########################################
        estimators['u_d_mu'] = None
        estimators['p_d_mu'] = None
        estimators['output_functional_hat_d_mus'] = None
        estimators['hessian_d_mu_il'] = None

        ##########################################

        # Functional hat d_mu 
        class output_hat_d_mu_RBEstimator(ImmutableObject):
            def __init__(self, primal_estimator, dual_estimator, cont_l_dmu, cont_a_dmu_functional,
                         primal_product, dual_product, key, index,
                         cont_a=None, coercivity_estimator=None):
                self.__auto_init(locals())

            def estimate_error(self, U, P, mu, U_d_mu=None, P_d_mu=None):
                primal_estimate = self.primal_estimator.estimate_error(U, mu)[0]
                dual_estimate = self.dual_estimator.estimate_error(U, P, mu)

                norm_P = np.sqrt(self.dual_product.apply2(P,P))[0,0]
                norm_U = np.sqrt(self.primal_product.apply2(U,U))[0,0]

                a_dmu_norm_estimate = self.cont_a_dmu_functional(mu)

                l_dmu_norm_estimate = self.cont_l_dmu(mu)

                est2 = primal_estimate *  (a_dmu_norm_estimate * norm_P) + \
                       dual_estimate * (l_dmu_norm_estimate + a_dmu_norm_estimate * norm_U) + \
                       primal_estimate * dual_estimate * a_dmu_norm_estimate
                return est2

        if self.prepare_for_gradient_estimate:
            if self.separated_bases and self.fom.adjoint_approach:
                assert 0, "not available"
            else:
                if self.adjoint_estimate:
                    assert 0, "not available"
                else:
                    print('GRAD J ESTIMATOR: non corrected estimator')
                    J_d_mu = self._construct_zero_dict(self.primal_fom.parameters)
                    for (key, size) in sorted(self.primal_fom.parameters.items()):
                        for l in range(size):
                            J_d_mu[key][l] = output_hat_d_mu_RBEstimator(estimators['primal'], estimators['dual'],
                                                                           self.cont_l_dmu[key][l], self.cont_a_dmu_functional[key][l],
                                                                           self.primal_rom.opt_product, self.dual_rom.opt_product,
                                                                           key, l)

                estimators['output_functional_hat_d_mus'] = J_d_mu
        ###############################################

        return estimators

    def project_output(self):
        output_functional = self.fom.output_functional_dict
        li_part = output_functional['linear_part']
        bi_part = output_functional['bilinear_part']
        d_u_li_part = output_functional['d_u_linear_part']
        d_u_bi_part = output_functional['d_u_bilinear_part']

        RB = self.RBPrimal
        projected_functionals = {
            'output_coefficient' : output_functional['output_coefficient'],
            'linear_part' : project(li_part, RB, None),
            'bilinear_part' : project(bi_part, RB, RB),
            'd_u_linear_part' : project(d_u_li_part, RB, None),
            'd_u_bilinear_part' : project(d_u_bi_part, RB, RB),
            'dual_projected_d_u_bilinear_part' : project(d_u_bi_part, RB, self.RBDual),
            'primal_dual_projected_op': project(self.fom.primal_model.operator, RB, self.RBDual),
            'dual_projected_rhs': project(self.fom.primal_model.rhs, self.RBDual, None),
            'primal_projected_dual_rhs': project(self.dual_intermediate_fom.rhs, RB, None),
        }
        return projected_functionals

    def project_adjoint_hessian(self):
        rom = self.primal_rom
        dual_rom = self.dual_rom
        output_functional = self.projected_output
        output_coefficient = output_functional['output_coefficient']
        linear_part = output_functional['d_u_linear_part']
        bilinear_part = output_functional['d_u_bilinear_part']
        projected_hessian = {}
        for (key, size) in sorted(self.primal_fom.parameters.items()):
            hessian = np.empty(size, dtype=dict)
            for l in range(size):
                P_D_op_no_d_mu = output_functional['primal_dual_projected_op']
                P_D_k_d_u_no_d_mu = output_functional['dual_projected_d_u_bilinear_part']

                #new
                _2_theta = output_coefficient.d_mu(key, l).d_mu(key, l)

                # preparations for the second loop
                D_rhs_ = output_functional['dual_projected_rhs'].d_mu(key,l)
                P_D_op_ = output_functional['primal_dual_projected_op'].d_mu(key,l)

                D_rhs = D_rhs_
                P_D_op = P_D_op_

                # new 
                P_P_op = rom.operator.d_mu(key,l)
                P_rhs = rom.rhs.d_mu(key,l)
                D_D_op = dual_rom.operator.d_mu(key,l)
                D_P_k = P_D_k_d_u_no_d_mu.H

                D_D_dual_op = dual_rom.operator.d_mu(key,l)

                P_D_k_d_u = P_D_k_d_u_no_d_mu

                proj_hessian = {
                    'D_rhs' : D_rhs,
                    'P_rhs' : P_rhs,
                    'P_D_op' : P_D_op,
                    'P_P_op': P_P_op,
                    'P_D_op': P_D_op,
                    'P_P_op': P_P_op,
                    'D_D_op': D_D_op,
                    'D_P_k': D_P_k,
                    'D_D_dual_op': D_D_dual_op,
                    'P_D_k_d_u': P_D_k_d_u,
                    'P_D_op_no_d_mu': P_D_op_no_d_mu,
                    '2_theta': _2_theta,
                    }
                hessian[l] = proj_hessian
            projected_hessian[key] = hessian
        return projected_hessian

    def project_hessian(self):
        RBP = self.RBPrimal
        RBD = self.RBDual
        fom = self.fom.primal_model
        output_functional = self.fom.output_functional_dict
        linear_part = output_functional['d_u_linear_part']
        bilinear_part = output_functional['d_u_bilinear_part']
        projected_hessian = {}
        for (key, size) in sorted(self.primal_fom.parameters.items()):
            hessian = np.empty(size, dtype=dict)
            for l in range(size):
                D_rhs = project(fom.rhs.d_mu(key,l), RBD, None)
                P_D_op = project(fom.operator.d_mu(key,l), RBP, RBD)
                proj_hessian = {
                    'D_rhs' : D_rhs,
                    'P_D_op' : PD_op,
                    }
                hessian[l] = proj_hessian
            projected_hessian[key] = hessian
        return projected_hessian

    def project_product(self):
        projected_product = project(self.opt_product, self.RBPrimal, self.RBPrimal)
        return projected_product

    def assemble_estimator_for_subbasis(self, dims):
        raise NotImplementedError

    def _reduce_to_subbasis(self, dims):
        raise NotImplementedError

    def _reduce_to_primal_subbasis(self, dim):
        raise NotImplementedError

class NonAssembledCoerciveRBReductor(StationaryRBReductor):
    def __init__(self, fom, RB=None, product=None, coercivity_estimator=None,
                 check_orthonormality=None, check_tol=None):
        assert fom.operator.linear and fom.rhs.linear
        assert isinstance(fom.operator, LincombOperator)
        assert all(not op.parametric for op in fom.operator.operators)
        if fom.rhs.parametric:
            assert isinstance(fom.rhs, LincombOperator)
            assert all(not op.parametric for op in fom.rhs.operators)

        super().__init__(fom, RB, product=product, check_orthonormality=check_orthonormality,
                         check_tol=check_tol)
        self.coercivity_estimator = coercivity_estimator

    def assemble_error_estimator(self):
        # compute the Riesz representative of (U, .)_L2 with respect to product

        class non_assembled_estimator(ImmutableObject):
            def __init__(self, fom, product, reductor):
                self.__auto_init(locals())
            def estimate_error(self, U, mu, m):
                U = self.reductor.reconstruct(U)
                riesz = self.product.apply_inverse(self.fom.operator.apply(U, mu) - self.fom.rhs.as_vector(mu))
                sqrt = self.product.apply2(riesz,riesz)
                output = np.sqrt(sqrt)
                return output
        return non_assembled_estimator(self.fom, self.products['RB'], self)

    def assemble_estimator_for_subbasis(self, dims):
        return self._last_rom.error_estimator.restricted_to_subbasis(dims['RB'], m=self._last_rom)


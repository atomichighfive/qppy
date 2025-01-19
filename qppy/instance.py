from typing import Optional, Union
from enum import Enum

import numpy as np
import numba as nb

class ConstraintRelation(Enum):
    LESS_THAN = 1
    EQUAL = 2
    GREATER_THAN = 3

class VariableDomain(Enum):
    REAL = 1
    BINARY = 2
    INTEGER = 3

class LinearConstraint:
    def __init__(self, name: str):
        assert isinstance(name, str), "Constraint name must be a string."
        self.name = name
        self._coefficients = {}
        self._relation = None
        self._rhs = None

    def add_term(self, variable: str, coefficient: float):
        assert variable not in self._coefficients, f"Variable {variable} already exists in constraint {self.name}."
        self._coefficients[variable] = coefficient

    def set_relation(self, relation: ConstraintRelation):
        assert self._relation is None, f"Relation already set for constraint {self.name}."
        self._relation = relation

    def set_rhs(self, rhs: float):
        assert self._rhs is None, f"RHS already set for constraint {self.name}."
        self._rhs = rhs

    def get_relation(self, standard_form: bool = True):
        if not standard_form:
            return self._relation
        elif self._relation == ConstraintRelation.LESS_THAN:
            return ConstraintRelation.LESS_THAN
        elif self._relation == ConstraintRelation.GREATER_THAN:
            return ConstraintRelation.LESS_THAN  # Convert to standard form means greater than becomes less than
        elif self._relation == ConstraintRelation.EQUAL:
            raise NotImplementedError(f"Equality constraints not supported but {self.name} is an equality constraint.")

    def get_coefficients(self, standard_form: bool = True):
        if self._relation == ConstraintRelation.LESS_THAN or not standard_form:
            return self._coefficients
        elif self._relation == ConstraintRelation.GREATER_THAN:
            # Convert to standard form by flipping the sign of the coefficients
            coefficients = {}
            for variable, coefficient in self._coefficients.items():
                coefficients[variable] = -coefficient
            return coefficients
        elif self._relation == ConstraintRelation.EQUAL:
            raise NotImplementedError(f"Equality constraints not supported but {self.name} is an equality constraint.")
    
    def get_rhs(self, standard_form: bool = True):
        if self._relation == ConstraintRelation.LESS_THAN or not standard_form:
            return self._rhs
        elif self._relation == ConstraintRelation.GREATER_THAN:
            # Convert to standard form by flipping the sign of the rhs
            return -self._rhs
        elif self._relation == ConstraintRelation.EQUAL:
            raise NotImplementedError(f"Equality constraints not supported but {self.name} is an equality constraint.")
    
class Variable:
    def __init__(self, name: str, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None, domain: VariableDomain = VariableDomain.REAL):
        assert isinstance(name, str), "Variable name must be a string."
        self.name = name
        self.domain = domain
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

class Instance:
    def __init__(self):
        self.minimize = True  # Whether to minimize or maximize the objective
        self.quadratic_objective_terms = {}  # Quadratic objective terms
        self.linear_objective_terms = {}  # Linear objective terms
        self._variables = {}  # Dict of all variables
        self._variable_index = {}  # Dict for quick lookup of variable index
        self._variable_lower_bound_constraint_index = {}  # Dict for quick lookup of variable lower bound index
        self._variable_upper_bound_constraint_index = {}  # Dict for quick lookup of variable upper bound index
        self._linear_constraints = {}  # Dict of all linear constraints
        self._linear_constraint_index = {}  # Dict for quick lookup of linear constraint index
        self._compiled = False  # Whether the instance has been compiled
        # These get populated when the instance is compiled
        self.Q = None  
        self.c = None
        self.A = None
        self.b = None

    @property
    def is_compiled(self):
        return self._compiled

    @property
    def variables(self):
        return self._variables
    
    @property
    def linear_constraints(self):
        return self._linear_constraints

    @property
    def _Q(self):
        n = len(self._variables)
        Q = np.zeros((n, n))
        for pair, coefficient in self.quadratic_objective_terms.items():
            pair = list(pair) if len(pair) == 2 else list(pair) + list(pair)  # Diagonal elements will have sets of len 1.
            i, j = self._variable_index[pair[0]], self._variable_index[pair[1]]
            if i == j:  # Multiply diagonals by two since we will be double counting off-diagonals in 0.5 * x.T @ Q @ x + c.T @ x, and then we divide by two.
                Q[i, j] = 2.0*coefficient if self.minimize else -2.0*coefficient
            else:
                Q[i, j] = coefficient if self.minimize else -coefficient
                Q[j, i] = coefficient if self.minimize else -coefficient
        return Q

    @property
    def _c(self):
        n = len(self._variables)
        c = np.zeros(n)
        for variable_name, coefficient in self.linear_objective_terms.items():
            c[self._variable_index[variable_name]] = coefficient if self.minimize else -coefficient
        c = c.reshape((n, 1))
        return c
    
    @property
    def _A_and_b(self):
        n = len(self._variables)
        m_inequality = len(self._linear_constraints)
        m_lower_bounds = 0
        m_upper_bounds = 0
        for variable in self._variables.values():
            if variable.lower_bound is not None:
                m_lower_bounds += 1
            if variable.upper_bound is not None:
                m_upper_bounds += 1
        
        A = np.zeros((m_inequality + m_lower_bounds + m_upper_bounds, n))
        b = np.zeros(m_inequality + m_lower_bounds + m_upper_bounds)

        for constraint in self.linear_constraints.values():
            for variable_name, coefficient in constraint.get_coefficients(standard_form = True).items():
                assert variable_name in self._variables, f"Variable {variable_name} of constraint {constraint.name} not in instance."
                A[
                    self._linear_constraint_index[constraint.name],
                    self._variable_index[variable_name]
                ] = coefficient
                b[self._linear_constraint_index[constraint.name]] = constraint.get_rhs(standard_form = True)
        
        for variable in self._variables.values():
            if variable.lower_bound is not None:
                A[
                    m_inequality + self._variable_lower_bound_constraint_index[variable.name],
                    self._variable_index[variable.name]
                ] = -1.0
                b[m_inequality + self._variable_lower_bound_constraint_index[variable.name]] = -variable.lower_bound
            if variable.upper_bound is not None:
                A[
                    m_inequality + m_lower_bounds + self._variable_upper_bound_constraint_index[variable.name],
                    self._variable_index[variable.name]
                ] = 1.0
                b[m_inequality + m_lower_bounds + self._variable_upper_bound_constraint_index[variable.name]] = variable.upper_bound
        b = b.reshape((-1, 1))
        return A, b
    
    def get_standard_form(self):
        self.assert_formulation()
        Q = self._Q
        c = self._c
        A, b = self._A_and_b
        return Q, c, A, b
    
    def assert_formulation(self):
        used_variables = set()
        for pair in self.quadratic_objective_terms.keys():
            if len(pair) == 2:
                v1, v2 = pair
                used_variables.add(v1)
                used_variables.add(v2)
            else:
                v = list(pair)[0]
                used_variables.add(v)
        for v in self.linear_objective_terms.keys():
            used_variables.add(v)
        unused_variables = set(self._variables.keys()) - used_variables
        assert len(unused_variables) == 0, f"Variables {unused_variables} are not used in the objective."
    
    def constraint_exists(self, constraint_name: str):
        return constraint_name in self._linear_constraints
    
    def variable_exists(self, variable_name: str):
        return variable_name in self._variables

    def set_minimize(self):
        assert not self._compiled, "Cannot change objective after compiling the instance."
        self.minimize = True

    def set_maximize(self):
        assert not self._compiled, "Cannot change objective after compiling the instance."
        self.minimize = False

    def add_variable(self, variable: Variable):
        assert not self._compiled, "Cannot add variables after compiling the instance."
        assert variable.name not in self._variables, f"Variable with name {variable.name} already exists."
        self._variable_index[variable.name] = len(self._variables)
        if variable.lower_bound is not None:
            self._variable_lower_bound_constraint_index[variable.name] = len(self._variable_lower_bound_constraint_index)
        if variable.upper_bound is not None:
            self._variable_upper_bound_constraint_index[variable.name] = len(self._variable_upper_bound_constraint_index)
        self._variables[variable.name] = variable

    def add_constraint(self, constraint: Union[LinearConstraint]):
        assert not self._compiled, "Cannot add constraints after compiling the instance."
        assert not self.constraint_exists(constraint_name=constraint.name), f"Constraint with name {constraint.name} already exists."
        self._linear_constraint_index[constraint.name] = len(self._linear_constraints)
        self._linear_constraints[constraint.name] = constraint

    def add_quadratic_objective_term(self, v1_name: str, v2_name: str, coefficient: float):
        assert not self._compiled, "Cannot add objective terms after compiling the instance."
        assert v1_name in self._variables, f"Variable {v1_name} not in instance"
        assert v2_name in self._variables, f"Variable {v2_name} not in instance"
        
        pair = frozenset([v1_name, v2_name])
        assert pair not in self.quadratic_objective_terms, f"Quadratic term for {pair} already exists."

        self.quadratic_objective_terms[pair] = coefficient

    def add_linear_objective_term(self, variable: str, coefficient: float):
        assert not self._compiled, "Cannot add objective terms after compiling the instance."
        assert variable in self._variables, f"Variable {variable} not in instance"
        assert variable not in self.linear_objective_terms, f"Linear term for {variable} already exists."

        self.linear_objective_terms[variable] = coefficient

    def solution_from_dict(self, solution_dict):
        x = np.zeros(len(self._variables))
        for variable_name, value in solution_dict.items():
            x[self._variable_index[variable_name]] = value
        x.reshape((-1, 1))
        x = np.ascontiguousarray(x)
        return x
    
    def vectorized_solution_from_dict(self, solution_dicts):
        X = [self.solution_from_dict(solution_dict) for solution_dict in solution_dicts]
        X = np.stack(X, axis=0)
        X = np.ascontiguousarray(X)
        return X

    def compile(self):
        assert not self._compiled, "Instance already compiled."
        self.assert_formulation()
        Q, c, A, b = self.get_standard_form()

        Q = np.ascontiguousarray(Q)
        c = np.ascontiguousarray(c)
        A = np.ascontiguousarray(A)
        b = np.ascontiguousarray(b)

        self.Q = Q
        self.c = c
        self.A = A
        self.b = b

        @staticmethod
        @nb.jit('f8(f8[::1])', nopython=True)
        def _objective(x: np.ndarray):
            return (0.5 * x.T @ Q @ x + c.T @ x)[0]
        self.objective = _objective

        @staticmethod
        def _vectorized_objective(X: np.ndarray):
            xTQx = np.einsum("ni,ij,nj->n", X, Q, X)
            cTx = np.einsum("ni,ij->n", X, c)
            return (0.5 * xTQx + cTx).flatten()
        self.vectorized_objective = _vectorized_objective

        @staticmethod
        @nb.jit('b1(f8[::1])', nopython=True)
        def _is_feasible(x: np.ndarray):
            return np.all(A @ x <= b.T)
        self.is_feasible = _is_feasible

        @staticmethod
        def _vectorized_is_feasible(X: np.ndarray):
            return np.all(np.einsum("ij,nj->ni", A, X) <= b.T[np.newaxis, :], axis=2).squeeze(axis=0)
        self.vectorized_is_feasible = _vectorized_is_feasible

        self._compiled = True
    
    
    
    def __str__(self):
        Q = self._Q
        A, b = self._A_and_b
        lines = []
        lines.append(f"qppy.Instance with {Q.shape[0]} variables and {A.shape[0]} constraints.")
        lines.append("minimize" if self.minimize else "maximize")
        if self.quadratic_objective_terms:
            lines.append("\t" + " ".join([f"{c:+}*{'*'.join(list(pair)) if len(pair) == 2 else list(pair)[0]+'²'}" for pair, c in self.quadratic_objective_terms.items()]))
        if self.linear_objective_terms:
            lines.append("\t" + " ".join([f"{c:+}*{v}" for v, c in self.linear_objective_terms.items()]))
        lines.append("subject to")
        for constraint in self.linear_constraints.values():
            match constraint.get_relation():
                case ConstraintRelation.LESS_THAN:
                    inequality = "<="
                case ConstraintRelation.EQUAL:
                    inequality = "=="
                case ConstraintRelation.GREATER_THAN:
                    inequality = ">="    
            lines.append(f"\t{constraint.name}: " + " + ".join([f"{c}*{v}" for v, c in constraint.get_coefficients().items()]) + f" {inequality} {constraint.get_rhs()}")
        for variable in self.variables.values():
            lines.append(f"\t{variable.name} in [{variable.lower_bound if variable.lower_bound is not None else 'inf'}, {variable.upper_bound if variable.upper_bound is not None else 'inf'}]")
        for variable in self.variables.values():
            match variable.domain:
                case VariableDomain.REAL:
                    lines.append(f"\t{variable.name} in R")
                case VariableDomain.BINARY:
                    lines.append(f"\t{variable.name} in {{0, 1}}")
                case VariableDomain.INTEGER:
                    lines.append(f"\t{variable.name} in Z")
        return "\n".join(lines)

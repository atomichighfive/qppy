import qppy as qp
import numpy as np

def test_qpinstance_correct_objective_value():
    I = qp.Instance()
    I.add_variable(qp.Variable('x', None, 0.9))
    I.add_quadratic_objective_term('x', 'x', 2.0)
    I.add_variable(qp.Variable('y', -0.4, 0.6))
    I.add_quadratic_objective_term('y', 'x', -3.0)
    I.add_linear_objective_term('y', -1.0)
    I.compile()

    def correct_func(x):
        return 2.0*x[0]**2 - 3.0*x[0]*x[1] - 1.0*x[1]
    
    np.random.seed(1)

    X = [
        I.solution_from_dict({'x': 1.0, 'y': np.random.uniform(-10.0, 10.0)})
        for _ in range(10000)
    ]
    for x in X:
        test = I.objective(x)
        correct = correct_func(x)
        assert np.isclose(test, correct, atol=1e-4), f"Failed for {x}. {test} != {correct}"

def test_qpinstance_correct_feasibility_check():
    I = qp.Instance()
    I.add_variable(qp.Variable('x', None, 0.9))
    I.add_quadratic_objective_term('x', 'x', 2.0)
    I.add_variable(qp.Variable('y', -0.4, 0.6))
    I.add_quadratic_objective_term('y', 'x', -3.0)
    I.add_linear_objective_term('y', -1.0)
    c1 = qp.LinearConstraint(name='c1')
    c1.add_term('x', 1.0)
    c1.add_term('y', 1.5)
    c1.set_relation(qp.ConstraintRelation.LESS_THAN)
    c1.set_rhs(0.5)
    I.add_constraint(c1)
    I.compile()

    def correct_check(x):
        return x[0] <= 0.9 and x[1] >= -0.4 and x[1] <= 0.6 and x[0] + 1.5*x[1] <= 0.5
    
    np.random.seed(1)
    X = [
        I.solution_from_dict({'x': 1.0, 'y': np.random.uniform(-10.0, 10.0)})
        for _ in range(10000)
    ]
    for x in X:
        test = I.is_feasible(x)
        correct = correct_check(x)
        assert test == correct, f"Failed for {x}. {test} != {correct}"


def test_qpinstance_vectorized_is_same():
    np.random.seed(1)
    
    for i_instance in range(100):
        N_variables = np.random.randint(1, 10)
        N_constraints = np.random.randint(1, 10)

        I = qp.Instance()
        for i_variable in range(0, N_variables):
            lower = np.random.uniform(-11.0, 10.0)
            upper = np.random.uniform(lower, 11.0)
            if lower < -10.0:
                lower = None
            if upper > 10.0:
                upper = None
            I.add_variable(qp.Variable(
                name=f'x_{i_variable}', 
                lower_bound=lower,
                upper_bound=upper,
                domain=qp.VariableDomain.REAL
            ))
        for i_constraint in range(0, N_constraints):
            c = qp.LinearConstraint(name=f'c_{i_constraint}')
            variables = [f'x_{i}' for i in set([np.random.randint(0, N_variables) for _ in range(0, np.random.randint(1, N_variables+1))])]
            for variable in variables:
                c.add_term(
                    variable=variable,
                    coefficient=np.random.uniform(-10.0, 10.0)
                )
            c.set_relation(np.random.choice(
                [
                    qp.ConstraintRelation.LESS_THAN,
                    #qp.ConstraintRelation.EQUAL,  # Not supported yet
                    qp.ConstraintRelation.GREATER_THAN
                ]
            ))
            c.set_rhs(np.random.uniform(-10.0, 10.0))
            I.add_constraint(c)

        all_variables = set([f'x_{i}' for i in range(0, N_variables)])
        used_variables = set()
        while len(all_variables - used_variables) > 0:
            if np.random.uniform() > 0.5:
                v1 = np.random.choice(list(all_variables - used_variables))
                v2 = np.random.choice(list(all_variables - used_variables))
                I.add_quadratic_objective_term(
                    v1_name=v1,
                    v2_name=v2,
                    coefficient=np.random.uniform(-10.0, 10.0)
                )
                used_variables.add(v1)
                used_variables.add(v2)
            else:
                v = np.random.choice(list(all_variables - used_variables))
                I.add_linear_objective_term(
                    variable=v,
                    coefficient=np.random.uniform(-10.0, 10.0)
                )
                used_variables.add(v)
        
        if np.random.uniform() > 0.5:
            I.set_maximize()
        
        I.compile()

        solution_dicts = [
            {
                f'x_{i}': np.random.uniform(-10.0, 10.0)
                for i in range(0, N_variables)
            }  for _ in range(1000)
        ]
        
        vectorised_x = I.vectorized_solution_from_dict(solution_dicts)
        vectorised_objective = I.vectorized_objective(vectorised_x)
        vectorised_feasibility = I.vectorized_is_feasible(vectorised_x)

        for i in range(0, len(solution_dicts)):
            x = I.solution_from_dict(solution_dicts[i])
            test_objective = I.objective(x)
            test_feasibility = I.is_feasible(x)
            assert np.isclose(vectorised_objective[i], test_objective, atol=1e-4), f"Failed for {x}. {vectorised_objective[i]} != {test_objective}. Instance was {I}"
            assert vectorised_feasibility[i] == test_feasibility, f"Failed for {x}. {vectorised_feasibility[i]} != {test_feasibility}. Instance was {I}"

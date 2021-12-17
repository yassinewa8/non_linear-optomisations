import sympy
import numpy as np
import sys


def input_txt():
    poly = '(x1 - 1)**2 + (x2 - 2)**2 +(x3 - 3)**2'                         #input for conjugate gradient
    poly_nomial = sympy.sympify(poly)
    global starting_point, epsilon
    epsilon = 10 ** -6
    starting_point = np.array([0.5, 0.5, 0.5])                              #State staring point
    return poly_nomial


def get_equation(x):
    return (x[0] - 1) ** 2 + (x[1] - 3) ** 2 + (x[2] - 3) ** 2              #input for steepest steepestdecsent


def write_output_to_textfile(output):
    f = open("steepestdecsentlog.txt", "w")
    f.write(output)


def values():
    poly_nomial = input_txt()
    #make a list of sympy variables based on the input poly_nomial
    variables = sympy.symbols(str(sorted(poly_nomial.free_symbols, key=lambda s: s.name)).replace(",", '')[1:-1])
    #print(variables)
    #generate hessian using input poly_nomial and variables
    hessian_matrix = np.array(sympy.hessian(poly_nomial, variables), dtype=np.float32)
    #print(hessian_matrix)
    #collecting linear values
    polynomial_list = [sympy.poly(poly_nomial, variable) for variable in variables]
    #put linear values into a list
    x = [val.coeffs()[1] for val in polynomial_list]
    #convert to numpy array
    lin_var = np.array(x, dtype=np.float32)
    py_func = sympy.lambdify(variables, poly_nomial)
    print(hessian_matrix, lin_var, py_func)
    return hessian_matrix, lin_var, py_func


def difference(X, Y):
    total_difference = 0
    for i in range(len(X)):
        total_difference = total_difference + abs(X[i] - Y[i])
    average_difference = total_difference / len(X)
    return average_difference

def derivate(f, X):
    derivates = []
    for i in range(len(X)):
        e = np.zeros(len(X))
        e[i] = epsilon
        values = X + e
        derivates.append((f(values) - f(X)) / epsilon)
    return derivates


def steepest_descent(X, epsilon, learning_rate=0.01):
    final_output = ""
    iterations = 0
    while True:
        iterations += 1
        d = derivate(get_equation, X)
        # print ("d :", d)
        x_prev = X
        X = X - np.dot(learning_rate, d)
        X = X.tolist()
        # print ("X :", X)
        # print ("x_prev :", x_prev)
        # print (difference(x_prev, X))
        final_output = final_output + "Iteration " + str(iterations) + " : " + str(difference(x_prev, X)) + "\n"
        if difference(x_prev, X) < epsilon:
            # print ("iterations :", iterations)
            write_output_to_textfile(final_output)
            return x_prev


def golden_section_search(function, initial, final):
    golden = (1 + 5 ** 0.5) / 2
    dif = final - initial
    x = final - dif / golden
    y = initial + dif / golden
    while np.abs(y - x) > epsilon:
        if function(x) < function(y):
            final = y
        else:
            initial = x
        x = final - (final - initial) / golden
        y = initial + (final - initial) / golden
    return (y + x) / 2


def conjugate_gradient(starting_point, epsilon, hessian_matrix, lin_var, py_func):
    # Calculate the initial gradient_vector
    gradient_vector = (hessian_matrix@starting_point + lin_var)
    directions_vector = -gradient_vector
    print(gradient_vector,directions_vector)
    # Initialise result array
    result_array = np.array([0., 0., 0.])
    # Iterations counter
    Iteration = 0
    #Initlising step arrays
    alphas_array = np.array([])
    betas_array = np.array([])
    direction_log = np.array(directions_vector)
    Result_vector = np.array(starting_point)
    while np.linalg.norm(directions_vector) > epsilon:
        if Iteration == 0:
            directions_vector = directions_vector
        else:
            Lima = gradient_vector.transpose()
            sigma = Lima * (gradient_vector - initial_gradient_vector)
            tango = initial_gradient_vector.transpose()
            omega = (tango * initial_gradient_vector)
            beta = (sigma / omega)
            betas_array = np.append(betas_array, beta)
            directions_vector = -gradient_vector + beta * directions_vector
        # calcuting apha using golden section search
        alpha = golden_section_search(lambda y: py_func(*result_array + y * directions_vector), 0, 1)
        # copy initial gradient_vector to use in next iteration
        initial_gradient_vector = gradient_vector.copy()
        # calculate new result_array
        result_array = result_array + alpha * directions_vector
        # Calculate gradient_vector
        gradient_vector = hessian_matrix @ result_array + lin_var
        # Save values to Iterations
        alphas_array = np.append(alphas_array, alpha)
        Iteration += 1
        #Stack resultant vectors
        Result_vector = np.vstack([Result_vector, result_array])
        #Stack direction vectors
        direction_log = np.vstack([direction_log, directions_vector])
    Iterations = np.array([Iteration, Result_vector, direction_log, alphas_array, betas_array], dtype='object')
    return result_array, Iterations


def Log_file(Iterations, filename='outs.txt'):
    Iteration, Result_vector, direction_log, alphas_array, betas_array = Iterations
    steps = ""
    Header =    f"//////////////////////////////////////Logs//////////////////////////////////////\n" \
                f"Iterations: {Iteration}\n" \
                f"Final Result: {Result_vector[-1]}"
    Initial =   f"\n//////////////////////////////////////////////////////////////////////////////////////////////////" \
                f"\n1)\nStarting Position {Result_vector[0]}\n" \
                f"directions_vector: {direction_log[0]}" \
                f"\nResult Vector X:{Result_vector[1]}"
    for i in range(2, Iteration + 1):
        if i < Iteration:
            steps += f"\n//////////////////////////////////////////////////////////////////////////////////////////////////" \
                         f"\n{str(i)})\nCurrent Value of Beta is: {betas_array[i-2]}," \
                         f"\ndirections_vector: {direction_log[i]}" \
                         f"\nPerforming Golden Section Search Alpha detected: {alphas_array[i-2]}" \
                         f"\nResult Vector X:{Result_vector[i]}\n" \
                         f"directions_vector normal is greater than epsilon\n" \
                         f"perfoming another iteration...."
        else:
            steps += f"\n//////////////////////////////////////////////////////////////////////////////////////////////////" \
                         f"\n{str(i)})\nCurrent Value of Beta is:{betas_array[i-2]}," \
                         f"\ndirections_vector: {direction_log[i]}" \
                         f"\nPerforming Golden Section Search Alpha detected: {alphas_array[-1]}" \
                         f"\nResult Vector X:{Result_vector[i]}\n" \
                         f"directions_vector normal is less than epsilon\n" \
                         f"Desired Solution Detected"
    output = Header + Initial + steps
    np.savetxt(filename, [output], delimiter=" ", fmt="%s")
    return None


def main():
    hessian_matrix, lin_var, py_func = values()
    solution, Iterations = conjugate_gradient(starting_point,epsilon,hessian_matrix, lin_var, py_func)
    #print(hessian_matrix)
    Log_file(Iterations, filename='ConjugateGradientLog.txt')
    output = open('solution.txt', 'w')
    sys.stdout = output
    output = steepest_descent(starting_point, epsilon)
    print("Conjugate gradient method result:", solution)
    print("Steepest Descent result:", output)
#def solutions(solution, Iterations):
if __name__ == '__main__':
    main()
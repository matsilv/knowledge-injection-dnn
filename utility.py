# Author: Mattia Silvestri

"""
    Utility script with methods and classes for the PLS problem.
"""

import numpy as np
import random
import sys
from ortools.sat.python import cp_model
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import math

########################################################################################################################


class PLSInstance:
    """
    Create an instance of the Partial Latin Square Constrained Problem with n numbers, one-hot encoded as
    an NxNxN numpy array. Each cell of the PLS represents a variable whose i-th bit is raised if number i is assigned.
    """
    def __init__(self, n=10, leave_columns_domains=False):
        # Problem dimension
        self.n = n
        # Problem instance
        self.square = np.zeros((n, n, n), dtype=np.int8)
        # Variables domains
        self._init_var_domains()
        self.remove_columns_domains = not leave_columns_domains

    def copy(self):
        """
        Create an instance which is equal to the current one.
        :return: PLSInstance.
        """

        obj = PLSInstance()
        obj.n = self.n
        obj.square = self.square.copy()
        obj.remove_columns_domains = self.remove_columns_domains
        return obj

    def _init_var_domains(self):
        """
        A method to initialize variables domains to [0, N]
        :return:
        """
        # (n, n) variables; n possible values; the index represents the value; 1 means removed from the domain
        self.domains = np.zeros(shape=(self.n, self.n, self.n), dtype=np.int8)

    def set_square(self, square, forward=False):
        """
        Public method to set the square.
        :param square: the square as numpy array of shape (dim, dim, dim)
        :param forward: True if you want to apply forward checking
        :return: True if the assignement is feasible, False otherwise
        """

        self.square = square
        feas = self._check_constraints()
        if feas and forward:
            self._init_var_domains()
            self._forward_checking()

        return feas

    def _check_constraints(self):
        """
        Check that all PLS constraints are consistent.
        :return: True if constraints are consistent, False otherwise.
        """
        multiple_var = np.sum(self.square, axis=2)
        rows_fail = np.sum(self.square, axis=1)
        cols_fail = np.sum(self.square, axis=0)

        if np.sum(multiple_var > 1) > 0:
            return False

        if np.sum(rows_fail > 1) > 0:
            return False

        if np.sum(cols_fail > 1) > 0:
            return False

        return True

    def check_constraints_type(self):
        """
        Check that all PLS constraints are consistent.
        :return: a list of three integers, one for each constraint type (multiple assignment, row violation,
        columns violation), representing violiations metric.
        """
        # How many times a value has been assigned to the same variable
        multiple_var = np.sum(self.square, axis=2)
        # How many times a value appears in the same row
        rows_fail = np.sum(self.square, axis=1)
        # How many times a value appears in the same columns
        cols_fail = np.sum(self.square, axis=0)

        constraint_type = [0, 0, 0]

        # How many times there is a multiple or lacking assignment
        constraint_type[0] = np.sum(multiple_var != 1)

        # How many equals values are there in a single row?
        constraint_type[1] = np.sum(rows_fail != 1)

        # How many equals values are there in a single column?
        constraint_type[2] = np.sum(cols_fail != 1)

        return constraint_type

    def get_assigned_variables(self):
        """
        Return indexes of assigned variables.
        :return: a numpy array containing indexes of assigned variables.
        """
        return np.argwhere(np.sum(self.square, axis=2) == 1)

    def _forward_checking(self):
        """
        Method to update variables domain with forward_checking
        :return:
        """

        for i in range(self.n):
            for j in range(self.n):
                # Find assigned value to current variable
                assigned_val = np.argwhere(self.square[i, j] == 1)
                assigned_val = assigned_val.reshape(-1)
                # Check if a variable is assigned
                if len(assigned_val) != 0:
                    # Current variable is already assigned -> domain is empty
                    self.domains[i, j] = np.ones(shape=(self.n,))
                    # Remove assigned value to same row and column variables domains
                    for id_cols in range(self.n):
                        self.domains[i, id_cols, assigned_val] = 1
                    if self.remove_columns_domains:
                        for id_row in range(self.n):
                            self.domains[id_row, j, assigned_val] = 1

    def assign(self, cell_x, cell_y, num):
        """
        Variable assignment.
        :param cell_x: x coordinate of a square cell
        :param cell_y: y coordinate of a square cell
        :param num: value assigned to cell
        :return: True if the assignment is consistent, False otherwise
        """

        # Create a temporary variable so that you can undo inconsistent assignment
        tmp_square = self.square.copy()

        if num > self.n-1 or num < 0:
            raise ValueError("Allowed values are in [0,{}]".format(self.n))
        else:
            self.square[cell_x, cell_y, num] += 1

        if not self._check_constraints():
            self.square = tmp_square.copy()
            return False

        return True

    def unassign(self, cell_x, cell_y):
        """
        Variable unassignment
        :param cell_x: x coordinate of a square cell
        :param cell_y: y coordinare of a square cell
        :return:
        """

        var = self.square[cell_x, cell_y].copy()
        assigned_val = np.argmax(var)
        self.square[cell_x, cell_y, assigned_val] = 0

    def visualize(self):
        """
        Visualize PLS.
        :return:
        """
        vals_square = np.argmax(self.square, axis=2) + np.sum(self.square, axis=2)

        print(vals_square)

########################################################################################################################


class PLSSolver:
    def __init__(self, board_size, square):
        """
        Class to build a PLS solver.
        :param board_size: number of variables
        :param square: numpy array with decimal assigned values
        """

        self.board_size = board_size

        # Create solver
        self.model = cp_model.CpModel()

        # Creates the variables.
        assigned = []
        for i in range(0, board_size ** 2):
            if square[i] > 0:
                assigned.append(self.model.NewIntVar(square[i], square[i], 'x%i' % i))
            else:
                assigned.append(self.model.NewIntVar(1, board_size, 'x%i' % i))

        # Creates the constraints.
        # All numbers in the same row must be different.
        for i in range(0, board_size ** 2, board_size):
            self.model.AddAllDifferent(assigned[i:i+board_size])

        # all numbers in the same column must be different
        for j in range(0, board_size):
            colmuns = []
            for idx in range(j, board_size ** 2, board_size):
                colmuns.append(assigned[idx])

            self.model.AddAllDifferent(colmuns)

        self.vars = assigned.copy()

    def solve(self):
        """
        Find a feasible solution.
        :return: True if a feasible solution was found, 0 otherwise
        """
        # create the solver
        solver = cp_model.CpSolver()
        # set time limit to 30 seconds
        solver.parameters.max_time_in_seconds = 30.0

        # solve the model
        status = solver.Solve(self.model)

        return status == cp_model.FEASIBLE


########################################################################################################################

def visualize(square):
    """
    Visualize PLS.
    :return:
    """
    vals_square = np.argmax(square, axis=2) + np.sum(square, axis=2)

    print(vals_square)

########################################################################################################################


def load_dataset(filename,
                 problem,
                 max_size=math.inf,
                 mode="onehot",
                 save_domains=False,
                 domains_filename=None,
                 save_partial_solutions=False,
                 partial_sols_filename=None,
                 assignments_filename=None):
    """
    Load solutions from a txt file in the PLS instance. It converts the legacy file format to the simpler CSV one,
    if save partial solution is specified.
    :param filename: name of the file; as string.
    :param problem: problem instance used to check feasibility; as PLSProblem.
    :param max_size: set max_size to prevent saturating the memory; as integer.
    :param mode: onehot, if you want to load a bit representation of variables; string,
                 if you want to load as string of 0-1; as string.
    :param save_domains: True if you want to compute variables domains and save them in a CSV file; as boolean.
    :param domains_filename: filename for variables domains; as string.
    :param save_partial_solutions: True if you want to save partial solutions in a CSV file; as boolean.
    :param partial_sols_filename: filename for partial solutions; as string.
    :param assignments_filename: filename for the assignments file; as string.
    :return: input instances and labels; as numpy array.
    """

    assert mode in ["onehot", "string"], "Unsupported mode"

    X = []
    Y = []

    with open(filename, mode="r") as file:
        domains_file = None

        if save_domains:
            domains_file = open(domains_filename, "w", newline='')
            csv_writer = csv.writer(domains_file, delimiter=',')

        if save_partial_solutions:
            partial_sols_file = open(partial_sols_filename, "w")
            csv_writer_sols = csv.writer(partial_sols_file, delimiter=',')
            assignments_file = open(assignments_filename, "w") 
            csv_writer_assignments = csv.writer(assignments_file, delimiter=',')

        dim = problem.n

        # Count number of solutions
        count = 0

        # Each line is the partial assignment and the successive assignment separated by "-" character
        while True and count < max_size:

            line = file.readline()
            if line is None or line is "":
                break

            solutions = line.split("-")

            # First element is the partial assignment, then the assignment
            label = False

            for sol in solutions:
                # Temporary problem instance
                tmp_problem = problem.copy()

                # Remove end of line
                if "\n" in sol:
                    sol = sol.replace("\n", "")

                # Check solution len is dim * dim * dim
                assert len(sol) == dim ** 3, "len is {}".format(len(sol))

                # dim * dim variables of dimension dim
                if mode == "onehot":

                    assignment = np.asarray(list(sol), dtype=np.int8)

                    # Check feasibility of loaded solutions
                    reshaped_assign = np.reshape(assignment, (dim, dim, dim))
                    if not label:
                        feasible = tmp_problem.set_square(reshaped_assign.copy(), not label and save_domains)
                        assert feasible, "Solution is not feasible"

                if not label:
                    if mode == "onehot":
                        X.append(assignment.copy())

                        if save_domains:
                            csv_writer.writerow(tmp_problem.domains.reshape(-1))
                        if save_partial_solutions:
                            csv_writer_sols.writerow(assignment.reshape(-1))
                    else:
                        X.append(sol)

                    # increase solutions counter
                    count += 1
                    if count % 10000 == 0:
                        print("Loaded {} examples".format(count))
                        print(
                            "Memory needed by X:{} | Memory needed by Y: {}".format(sys.getsizeof(X), sys.getsizeof(Y)))
                else:
                    if mode == "onehot":
                        Y.append(assignment.copy())
                        if save_partial_solutions:
                            csv_writer_assignments.writerow([np.argmax(assignment.reshape(-1))])
                    else:
                        Y.append(sol)

                # First element is the partial assignment, then the assignment
                label = True

        file.close()

        if save_domains:
            domains_file.close()
        if save_partial_solutions:
            partial_sols_file.close()

        # Check assignment is feasible
        if mode == "onehot":
            prob = PLSInstance()
            for x, y in zip(X, Y):
                square = np.reshape((x + y), (dim, dim, dim))
                assert prob.set_square(square), "Assignment is not feasible"

        # Return a numpy array
        X = np.asarray(X)
        Y = np.asarray(Y)

        X = X.reshape(X.shape[0], -1)
        Y = Y.reshape(Y.shape[0], -1)

        print("Memory needed by X:{} | Memory needed by Y: {}".format(sys.getsizeof(X), sys.getsizeof(Y)))

        return X, Y


########################################################################################################################


def random_assigner(dim, domains):
    """
    Return a random assignment
    :param dim: one-hot encoding  dimension of the assignment problem
    :param domains: variables' domains coming from forward checking as numpy array of shape (batch_size, 10, 10, 10)
    :return: assigned variable index
    """
    if domains is None:
        return random.randint(0, dim-1)
    else:
        allowed_assignments = np.argwhere(domains == 0)
        idx = random.randint(0, len(allowed_assignments)-1)
        return allowed_assignments[idx]

########################################################################################################################


class VarArraySolutionPrinterWithLimit(cp_model.CpSolverSolutionCallback):
    """Print and save  solutions."""

    def __init__(self, variables, limit):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_limit = limit
        self.solutions = []

    def on_solution_callback(self):
        """
        Invoked each time a solution is found.
        :return:
        """
        self.__solution_count += 1
        sol = np.zeros(shape=(100, ), dtype=np.int8)
        i = 0
        for v in self.__variables:
            sol[i] = self.Value(v)
            i += 1

        if len(self.solutions) >= self.__solution_limit:
            print('Stop search after %i solutions' % self.__solution_limit)
            self.StopSearch()

        if self.solution_count() % 1000 == 0:
            self.solutions.append(sol)
            print(len(self.solutions))

    def solution_count(self):
        """
        Accessor to count of solutions.
        :return:
        """
        return self.__solution_count

########################################################################################################################


def from_decimal_to_one_hot(int_array):
    """
    Convert an array of decimal number to a one-hot encoding.
    :param int_array: array of decimal number to be converted
    :return: numpy array of size 10
    """

    one_hot = np.zeros((int_array.size, int_array.max()), dtype=np.int8)
    one_hot[np.arange(int_array.size), int_array-1] = 1

    return one_hot.reshape(-1)

########################################################################################################################


def read_solutions_from_csv(filename, dim, max_size=100000):
    """
    Method to read solutions from a csv file and check if it is feasible and which constraints are not satisfied
    :param filename: name of the file from which to read the solutions
    :param dim: problem dimension; as integer
    :param max_size: maximum number of solutions to be loaded
    :return:
    """
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        # Count of feasible solutions
        count_feasible_sols = 0
        # Count single assignment violations
        count_single_assign_violations = 0
        # Count of rows constraint violations
        count_rows_violations = 0
        # Count of columns constraint violations
        count_columns_violations = 0
        # Count of examined solutions
        count_solutions = 0

        for line in csv_reader:
            if len(line) != dim ** 3:
              continue
            assert len(line) == dim**3

            one_hot_sol = [int(c) for c in line]
            one_hot_sol = np.asarray(one_hot_sol)
            reshaped_sol = np.reshape(one_hot_sol, (dim, dim, dim))

            problem = PLSInstance()
            problem.set_square(reshaped_sol.copy())

            constraints_satisfied = problem.check_constraints_type()
            count_solutions += 1
            count_single_assign_violations += constraints_satisfied[0]
            count_rows_violations += constraints_satisfied[1]
            count_columns_violations += constraints_satisfied[2]
            if constraints_satisfied[0] == 0 and constraints_satisfied[1] == 0 and constraints_satisfied[2] == 0:
                count_feasible_sols += 1

            if count_solutions == max_size:
                break

        print("Count of feasible solutions: {}".format(count_feasible_sols))
        print("Count of solutions: {}".format(count_solutions))
        print("Count of single variable assignment violations: {}".format(count_single_assign_violations))
        print("Count of rows constraint violations: {}".format(count_rows_violations))
        print("Count of columns constraint violations: {}".format(count_columns_violations))


########################################################################################################################


def compute_feasibility_from_predictions(X, preds, dim):
    """
    Given partial assignments, compute feasibility of network predictions.
    :param X: partial assignment; as numpy array.
    :param preds: network predictions; as numpy array.
    :param dim: PLS dimension; as integer.
    :return: float; feasibility value.
    """

    feas_count = 0

    for x, pred in zip(X, preds):
        # Create a problem instance with current training example for net prediction
        square = np.reshape(x, (dim, dim, dim))
        pls = PLSInstance(n=dim)
        pls.square = square.copy()
        # assert pls.__check_constraints__(), "Constraints should be verified before assignment"

        # Make the prediction assignment
        assignment = np.argmax(pred)
        assignment = np.unravel_index(assignment, shape=(dim, dim, dim))

        # Local consistency
        local_feas = pls.assign(assignment[0], assignment[1], assignment[2])

        # Global consistency
        if local_feas:
            vals_square = np.argmax(pls.square.copy(), axis=2) + np.sum(pls.square.copy(), axis=2)
            solver = PLSSolver(dim, square=np.reshape(vals_square, -1), specialized=False,
                                       size=0)
            feas = solver.solve()
        else:
            feas = local_feas

        if feas:
            feas_count += 1

    return feas_count / X.shape[0]

########################################################################################################################


def make_subplots(nested_path, n_subplots, labels, titles, pls_sizes=[7, 10, 12]):
    """
    Method to make 'feasibility plots'.
    :param nested_path: list of lists of strings; where the feasibility results are loaded from.
    :param n_subplots: int; number of subplots (e.g. comparing problem dimensions and lambdas values.
    :param labels: list of string; labels for the curve (same for each subplot).
    :param titles: list of string; name of the subplots.
    :param pls_sizes: list of int; size of the PLS for each subplot.
    :return:
    """
    sns.set_style('darkgrid')
    linestyles = ['solid', 'solid', 'dotted', 'dashed', 'dashdot', (0, (3, 1, 1, 1, 1, 1)), (0, (5, 10))]
    assert len(nested_path) == n_subplots
    plt.rcParams["figure.figsize"] = (20, 15)
    fig, axis = plt.subplots(1, n_subplots, sharey=True)
    fig.text(0.5, 0.04, '# of filled cells', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.07, 0.5, 'Feasibility ratio', va='center', rotation='vertical', fontsize=16, fontweight='bold')
    subplots = [*axis]
    for subp_idx, (paths, subp) in enumerate(zip(nested_path, subplots)):
        ax = subplots[subp_idx]
        ax.set_xlim(0, pls_sizes[subp_idx]**2)
        ax.set_ylim(0, 1.1)
        ticks = np.arange(0, pls_sizes[subp_idx]**2, pls_sizes[subp_idx]*2)
        assert len(paths) <= len(linestyles), 'Max {} curves can be plotted in the same subplot'.format(len(linestyles))
        ax.set_xticks(ticks)
        ax.tick_params(labelsize=14)
        for path_idx, path in enumerate(paths):
            feasibilities = np.genfromtxt(path, delimiter=',')
            ax.plot(np.arange(len(feasibilities)), feasibilities, linestyle=linestyles[path_idx])
        if subp_idx == 0:
            ax.legend(labels, fontsize=16, loc='lower left')

        ax.set_title(titles[subp_idx], fontweight='bold', fontsize='18')
    plt.show()

########################################################################################################################


if __name__ == '__main__':
    # This is an example of how to visualize constraints violations
    read_solutions_from_csv(filename='solutions/pls12/model_agnostic_100k-sols_no_prop.csv', dim=12, max_size=100000)
    print()
    read_solutions_from_csv(filename='solutions/pls12/sbr_full_all_ts_no_prop.csv', dim=12, max_size=100000)
    print()
    read_solutions_from_csv(filename='solutions/pls12/sbr_full_100k-sols_no_prop.csv', dim=12, max_size=100000)

    # This is an example on how to visualize 'feasibility plots'
    paths = [['plots/test-pls-7-tf-keras/random/rows-prop/random_feasibility.csv',
              'plots/test-pls-7-tf-keras/random/rows-and-columns-prop/random_feasibility.csv',
              'plots/test-pls-7-tf-keras/model-agnostic/100-sols/run-1/feasibility_test.csv',
              'plots/test-pls-7-tf-keras/sbr-inspired-loss/100-sols/rows/run-1/feasibility_test.csv',
              'plots/test-pls-7-tf-keras/sbr-inspired-loss/100-sols/full/run-1/feasibility_test.csv'],

             ['plots/test-pls-10-tf-keras/random/rows-prop/random_feasibility.csv',
              'plots/test-pls-10-tf-keras/random/rows-and-columns-prop/random_feasibility.csv',
              'plots/test-pls-10-tf-keras/model-agnostic/100-sols/run-1/feasibility_test.csv',
              'plots/test-pls-10-tf-keras/sbr-inspired-loss/100-sols/rows/run-1/feasibility_test.csv',
              'plots/test-pls-10-tf-keras/sbr-inspired-loss/100-sols/full/run-1/feasibility_test.csv'],

             ['plots/test-pls-12-tf-keras/random/rows-prop/random_feasibility.csv',
              'plots/test-pls-12-tf-keras/random/rows-and-columns-prop/random_feasibility.csv',
              'plots/test-pls-12-tf-keras/model-agnostic/100-sols/run-1/feasibility_test.csv',
              'plots/test-pls-12-tf-keras/sbr-inspired-loss/100-sols/rows/run-1/feasibility_test.csv',
              'plots/test-pls-12-tf-keras/sbr-inspired-loss/100-sols/full/run-1/feasibility_test.csv']]

    lbls = ['rnd-rows', 'rnd-full', 'agn', 'mse-rows', 'mse-full']
    tls = ['PLS-7', 'PLS-10', 'PLS-12']

    make_subplots(paths, n_subplots=3, labels=lbls, titles=tls, pls_sizes=[7, 10, 12])


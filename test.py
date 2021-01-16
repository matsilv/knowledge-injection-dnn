import utility
import time
import numpy as np

X = utility.load_dataset_with_tf("datasets/pls7/partial_solutions_10k.csv", column_types=[int() for _ in range(7**3)])
Y = utility.load_dataset_with_tf("datasets/pls7/assignments_10k.csv", column_types=[int() for _ in range(7**3)])
X = X.shuffle(buffer_size=1000, seed=1).batch(16)
Y = Y.shuffle(buffer_size=1000, seed=1).batch(16)

for x, y in zip(X, Y):
    x = x.numpy()
    y = y.numpy()

    x = x[0]
    y = y[0]

    utility.visualize(x.reshape(7, 7, 7))
    print()
    utility.visualize(y.reshape(7, 7, 7))
    print('-' * 20)


exit()

########################################################################################################################
DIM = 10
SOL_TYPE = "UNIQUES"
mode_char = "B"
NUM_SOL = "10k"
file_name = "pls{}_{}".format(DIM, NUM_SOL)
LEAVE_COLUMNS_DOMAINS = True
MAX_SIZE = 100
LOAD_MODE = "onehot"
SAVE_DOMAINS = False
mode = "train"
domains_filename = "datasets/pls{}/domains_{}_{}.csv".format(DIM, mode, NUM_SOL)
load_path = "datasets/pls{}/DS.PLS.A.{}.{}.4.{}.txt".format(DIM, SOL_TYPE, mode_char, file_name)
########################################################################################################################

# Create problem instance
problem = utility.PLSInstance(n=DIM, leave_columns_domains=LEAVE_COLUMNS_DOMAINS)
init_prob = problem.copy()
print("Loading data from {}".format(load_path))
start_time = time.time()

X, Y = utility.load_dataset(load_path, problem, max_size=MAX_SIZE, mode=LOAD_MODE, save_domains=SAVE_DOMAINS,
                            domains_filename=domains_filename, save_partial_solutions=False,
                            partial_sols_filename="datasets/pls10/partial_sols_pls10_num_assigned_40.csv")
end_time = time.time()
print("Elapsed {} sec".format(end_time - start_time))

start_time = time.time()
utility.find_feasible_and_unfeasible_assignments(X, DIM)
end_time = time.time()
print("Elapsed {} sec".format(end_time - start_time))
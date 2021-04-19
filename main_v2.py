# @author: Mattia Silvestri

"""
Main program.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from utility import PLSInstance, PLSSolver, random_assigner
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import argparse
import time
from models import MyModel
import pandas as pd

########################################################################################################################

# Set seed in order to reproduce results
tf.random.set_seed(0)

# Tensorflow 2 GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

########################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=10, help="Problem dimension")
parser.add_argument("--train", action="store_true",
                    help="Train the model; if not set the default is test mode", default=False)
parser.add_argument("--spec", action="store_true",
                    help="Use model trained with an additional specializing constraint", default=False)
parser.add_argument("--spec-constr", action="store_true",
                    help="Additional specializing constraint", default=False)
parser.add_argument("--size", type=int, default=0,
                    help="Indexes size of the additional specializing contraint")
parser.add_argument("--original", action="store_true",
                    help="Original dataset; otherwise the dataset with an additional specializing " +
                         "constraint is used", default=True)
parser.add_argument("--test-num", default=None,
                    help="Test identifier; view param.doc file for more details")
parser.add_argument("--num-epochs", default=300, type=int,
                    help="Number of training epochs")
parser.add_argument("--max-size", default=1000000, type=int,
                    help="Maximum number of dataset size to be loaded")
parser.add_argument("--visualize", default=False, action="store_true",
                    help="Visualize solutions and assignments frequencies")
parser.add_argument("--load-mode", default="onehot", choices=["onehot", "string"],
                    help="Dataset loading mode")
parser.add_argument("--batch-size", default=1024, type=int,
                    help="Mini-batch size")
parser.add_argument("--save-domains", action="store_true", default=False,
                    help="Compute variables domains with forward checking propagator and save them in a CSV file")
parser.add_argument("--leave-columns-domains", action="store_true", default=False,
                    help="True if you don't want to prune columns domains values with fowrad checking")
parser.add_argument("--multiple", action="store_true", default=False,
                    help="Set this flag if you want to use the dataset coming from multiple random deconstruction "
                         + "of the same solutions pool")
parser.add_argument("--num-sol", type=str, default="10k",
                    help="Number of solutions from which the training set has been generated; thousands are expressed "
                         + "with k (for example 10000=10k)")
parser.add_argument("--penalties-type", default=None, choices=["multi", "domains"],
                    help="Penalties type to be used both for training and test")
parser.add_argument("--model-type", default="agnostic", choices=["agnostic", "sbrinspiredloss", "negative", "binary"],
                    help="Choose the model type. agnostic is a simple Sequential model full agnostic. sbrinspiredloss "
                         + "envelops the SBR-inspired loss function.")
parser.add_argument("--validation-size", type=int, default=0,
                    help="Validation set dimension. If zero no validation set is used")

parser.add_argument("--use-prop", action="store_true", default=False,
                    help="True if you want to assist the with propagation method during evaluation time." +
                         "A penalty type must be assigned.")
parser.add_argument("--rnd-feas", action="store_true", default=False,
                    help="True if you want to compute feasibility ratio also for random assigner")

parser.add_argument("--lmbd", default=1.0, type=float, help="Lambda for SBR-inspired term")

parser.add_argument("--patience", default=5, type=int,
                    help="Specify the number of 10 epochs intervals without improvment in "
                         "feasibility after which training is stopped")

args = parser.parse_args()
########################################################################################################################

# Problem dimension
DIM = int(args.dim)

COLUMN_TYPES = [int() for _ in range(DIM**3)]

# Set training or test mode
TRAIN = args.train
if TRAIN:
    print("Training mode")
    mode = "train"
else:
    print("Test mode")
    mode = "test"

# Flag to specify which kind of dataset you want to use: the original one, or the one with additional specializing
# constraint
SPECIALIZED = args.spec

# True if you want to check the additional specializing constraint
SPECIALIZED_CONSTRAINT = args.spec_constr

# Number of additional specialized constraint indexes to consider
SIZE = int(args.size)

ORIGINAL = args.original

# Test number identifier
TEST_NUM = args.test_num

# Number of training epochs
EPOCHS = int(args.num_epochs)

# Maximum number of data set examples to load
MAX_SIZE = int(args.max_size)

# Visualize partial and successive assignments frequencies in the data set
VISUALIZE = args.visualize

# Available loading mode are string and one-hot
LOAD_MODE = args.load_mode

# Mini-batch size
BATCH_SIZE = int(args.batch_size)

# True if you want to adopt SRB-inspired loss function
MODEL_TYPE = args.model_type

# True if you want to save pruned variables domains in a CSV file
SAVE_DOMAINS = args.save_domains

# Multiple random deconstrucion of solutions or uniques partial assignments
if args.multiple:
    SOL_TYPE = "MULTIPLE"
else:
    SOL_TYPE = "UNIQUES"

if SPECIALIZED_CONSTRAINT:
    print("Specialized constraint")
else:
    print("Standard constraint")

# label for plotting
if SPECIALIZED:
    print("Specialized model")
    label_name = "specialized"
else:
    print("Original model")
    label_name = "original"
label = "net trained on {} ds".format(label_name)

if mode == "test":
    mode_char = "L"
    SOL_TYPE = "UNIQUES"
else:
    mode_char = "B"

if ORIGINAL:
    if not TRAIN:
        file_name = "pls{}_10k".format(DIM)
    else:
        file_name = "pls{}_{}".format(DIM, args.num_sol)
else:
    file_name = "pls{}_checker_rule_size_{}".format(DIM, SIZE)

VAL_SIZE = args.validation_size

NUM_SOL = args.num_sol

# Model name for both training and test
if not SPECIALIZED:
    model_name = "original_dataset/test-{}/".format(TEST_NUM)
else:
    model_name = "specialized_solutions/test-{}/".format(TEST_NUM)

# Where to save plots
SAVE_PATH = "plots/test-{}/".format(TEST_NUM)
try:
    os.makedirs(SAVE_PATH)
except:
    print("Directory already exists")

# Model name for both training and test
if not SPECIALIZED:
    model_name = "original_dataset/test-{}/".format(TEST_NUM)
else:
    model_name = "specialized_solutions/test-{}/".format(TEST_NUM)

# Where to save plots
SAVE_PATH = "plots/test-{}/".format(TEST_NUM)
try:
    os.makedirs(SAVE_PATH)
except:
    print("Directory already exists")

########################################################################################################################

# Create a validation set if required
val_indexes = None

if VAL_SIZE > 0:
    print("Loading validation set...")
    start = time.time()
    X_val = pd.read_csv("datasets/pls{}/partial_solutions_{}_train.csv".format(DIM, NUM_SOL),
                        sep=',',
                        header=None,
                        nrows=MAX_SIZE,
                        dtype=np.int8).values

    # Create penalties for the examples
    if MODEL_TYPE != 'agnostic':
        P_val = pd.read_csv("datasets/pls{}/domains_train_{}.csv".format(DIM, NUM_SOL),
                            sep=',',
                            header=None,
                            nrows=MAX_SIZE,
                            dtype=np.int8).values
    else:
        P_val = np.zeros_like(X_val, dtype=np.int8)

    end = time.time()
    print("Elapsed {} seconds".format((end - start)))

    val_indexes = np.random.choice(np.arange(0, X_val.shape[0]), size=VAL_SIZE, replace=False)
    X_val = X_val[val_indexes]
    P_val = P_val[val_indexes]
    validation_set = (X_val, P_val)

# Load training examples
features_filepath = "datasets/pls{}/partial_solutions_{}_{}.csv".format(DIM, NUM_SOL, mode)
print("Loading features from {}...".format(features_filepath))
start = time.time()
X = pd.read_csv(features_filepath, sep=',', header=None, nrows=MAX_SIZE, dtype=np.int8).values
end = time.time()
print("Elapsed {} seconds, {} GB required".format((end - start), X.nbytes / 10**9))
print("Number of rows: {}".format(X.shape[0]))

labels_filepath = "datasets/pls{}/assignments_{}_{}.csv".format(DIM, NUM_SOL, mode)
print("Loading labels from {}...".format(labels_filepath))
start = time.time()
Y = pd.read_csv(labels_filepath, sep=',', header=None, nrows=MAX_SIZE, dtype=np.int32).values
end = time.time()
print("Elapsed {} seconds, {} GB required".format((end - start), Y.nbytes / 10**9))

# Create penalties for the examples
if MODEL_TYPE == 'agnostic' and not args.use_prop:
    P = np.zeros_like(X, dtype=np.int8)
else:
    if not args.leave_columns_domains:
        penalties_filepath = "datasets/pls{}/domains_{}_{}.csv".format(DIM, mode, NUM_SOL)
    else:
        penalties_filepath = "datasets/pls{}/rows_propagation_domains_{}_{}.csv".format(DIM, mode, NUM_SOL)

    print("Loading penalties from {}...".format(penalties_filepath))
    start = time.time()
    P = pd.read_csv(penalties_filepath, sep=',', header=None, nrows=MAX_SIZE, dtype=np.int8).values
end = time.time()
print("Elapsed {} seconds, {} GB required".format((end - start), P.nbytes / 10**9))

# Remove validation samples from the training set
if val_indexes is not None:
    X = np.delete(X, val_indexes, axis=0)
    Y = np.delete(Y, val_indexes, axis=0)
    P = np.delete(P, val_indexes, axis=0)

# Create TF datasets
dataset = tf.data.Dataset.from_tensor_slices((X, Y, P)).shuffle(10000).batch(BATCH_SIZE)

# Create the model
model = MyModel(num_layers=2,
                num_hidden=[512, 512],
                input_shape=X.shape[1:],
                output_dim=DIM ** 3,
                method=MODEL_TYPE,
                lmbd=args.lmbd)

# Train model
if TRAIN:
    history = model.train(EPOCHS,
                          dataset,
                          "models/{}".format(model_name),
                          DIM,
                          validation_set,
                          args.use_prop,
                          args.patience)

    for name in history.keys():
        values = history[name]

        plt.plot(np.arange(0, len(values)), values,
                 label=name)
        plt.ylim(bottom=0)
        plt.legend()
        plt.savefig("{}/{}.png".format(SAVE_PATH, name))
        plt.close()

        with open("{}/{}.csv".format(SAVE_PATH, name), "w") as file:
            wr = csv.writer(file)
            wr.writerow(values)
            file.close()
    exit(0)

else:
     model.model = tf.saved_model.load("models/{}".format(model_name))

################################################################################

# Test the model

# Make predictions
tensor_X = X.astype(np.float32)
predict_val = model.predict_from_saved_model(tensor_X).numpy()

if args.use_prop:
    predict_val *= (1 - P)

# Count of correct predictions grouped by number of assigned variables
pred_by_num_assigned = np.zeros(shape=(DIM ** 2))
# Count of feasible solutions grouped by number of assigned variables
feas_by_num_assigned = np.zeros(shape=(DIM ** 2))
# Count of total examples grouped by number of assigned variables
tot_by_num_assigned = np.zeros(shape=(DIM ** 2))
# Count of random correct predictions grouped by number of assigned variables
rand_pred_by_num_assigned = np.zeros(shape=(DIM ** 2))
# Count of random feasible solutions grouped by number of assigned variables
rand_feas_by_num_assigned = np.zeros(shape=(DIM ** 2))

# Compute overall accuracy on training set
acc = 0
count = 0
acc_rand = 0

# Compute accuracy grouped by number of assigned variables
preds = []
for x, pred, y, d in zip(X, predict_val, Y, P):

    if count % 1000 == 0:
        print("Examined {} instances".format(count))

    '''utility.visualize(x.reshape(DIM, DIM, DIM))
    print()
    utility.visualize(y.reshape(DIM, DIM, DIM))
    print()
    for i in range(DIM):
      for j in range(DIM):
        print(d.reshape(DIM, DIM, DIM)[i, j])
      print()
    print('------------------------------------------------------------------')'''

    num_assigned_vars = np.sum(x.astype(np.int8))
    pred_label = np.argmax(pred.reshape(-1))
    correct_label = np.argmax(y.reshape(-1))

    if pred_label == correct_label:
        acc += 1
        pred_by_num_assigned[num_assigned_vars] += 1

    # Create a problem instance with current examples for net prediction
    square = np.reshape(x, (DIM, DIM, DIM))
    pls = PLSInstance(n=DIM)
    pls.square = square.copy()
    # assert pls.__check_constraints__(), "Constraints should be verified before assignment"

    # Make the prediction assignment
    assignment = np.argmax(pred)
    assignment = np.unravel_index(assignment, shape=(DIM, DIM, DIM))

    # Local consistency
    local_feas = pls.assign(assignment[0], assignment[1], assignment[2])

    '''vals_square = np.argmax(square, axis=2) + np.sum(square, axis=2)
    solver = utility.PLSSolver(DIM, square=np.reshape(vals_square, -1))
    res = solver.solve()
    assert res, "Constraint solver is wrong because the input comes from a real solution"'''

    # Global consistency
    if local_feas:
        vals_square = np.argmax(pls.square.copy(), axis=2) + np.sum(pls.square.copy(), axis=2)
        solver = PLSSolver(DIM,
                                   square=np.reshape(vals_square, -1),
                                   specialized=SPECIALIZED_CONSTRAINT,
                                   size=SIZE)
        feas = solver.solve()
    else:
        feas = local_feas

    if feas:
        feas_by_num_assigned[num_assigned_vars] += 1

    ####################################################################################################################

    if args.rnd_feas:
        # check random assignment performance
        if not args.use_prop:
            d = None
        rand_assignment = random_assigner(DIM ** 3, d)
        if rand_assignment == correct_label:
            acc_rand += 1
            rand_pred_by_num_assigned[num_assigned_vars] += 1

        # create a problem instance with current training example for random prediction
        square = np.reshape(x, (DIM, DIM, DIM))
        pls = PLSInstance(n=DIM)
        pls.square = square.copy()
        #assert pls.__check_constraints__(), "Constraints should be verified before assignment"

        # make the random assignment
        rand_assignment = np.unravel_index(rand_assignment, shape=(DIM, DIM, DIM))

        local_feas = pls.assign(rand_assignment[0], rand_assignment[1], rand_assignment[2])

        # global consistency
        if local_feas:
            vals_square = np.argmax(pls.square.copy(), axis=2) + np.sum(pls.square.copy(), axis=2)
            solver = PLSSolver(DIM,
                                       square=np.reshape(vals_square, -1),
                                       specialized=SPECIALIZED_CONSTRAINT,
                                       size=SIZE)
            feas = solver.solve()
        else:
            feas = local_feas

        if feas:
            rand_feas_by_num_assigned[num_assigned_vars] += 1
    ####################################################################################################################

    '''print("Assignment: (x:{},y:{},val:{}) | Feasible: {}".format(assignment[0], assignment[1], assignment[2]+1, feas))
    pls.visualize()
    _ = input("Press to continue...")'''

    # Increase count of solutions with this number of assignments
    tot_by_num_assigned[num_assigned_vars] += 1
    count += 1

    if count % 1000 == 0:
      
        feasibility = list((feas_by_num_assigned / (tot_by_num_assigned + 1e-8))[1:])

        # Feasibility plot
        plt.plot(np.arange(1, DIM ** 2), feasibility, label=label)
        plt.ylim((0.0, 1.1))
        plt.legend()
        plt.savefig("feasibility_{}.png".format(mode))
        plt.close()

        if not args.use_prop:
            filename = "{}/feasibility_{}.csv".format(SAVE_PATH, mode)
        else:
            if args.leave_columns_domains:
                filename = "{}/feasibility_{}_with_row_prop.csv".format(SAVE_PATH, mode)
            else:
                filename = "{}/feasibility_{}_with_full_prop.csv".format(SAVE_PATH, mode)

        with open(filename, "w") as epoch_file:
            wr = csv.writer(epoch_file)
            wr.writerow(feasibility)

# Check accuracy is correctly computed
assert np.sum(pred_by_num_assigned) == acc and np.sum(tot_by_num_assigned) == count, \
    "acc: {} | acc_vectorized: {} | count: {} | count_vectorized: {}".format(acc, np.sum(pred_by_num_assigned),
                                                                             count, np.sum(tot_by_num_assigned))

########################################################################################################################

# Make plots

accuracy = list((pred_by_num_assigned / (tot_by_num_assigned + 1e-8))[1:])
feasibility = list((feas_by_num_assigned / (tot_by_num_assigned + 1e-8))[1:])
if args.rnd_feas:
    random_feasibility = list((rand_feas_by_num_assigned / (tot_by_num_assigned + 1e-8))[1:])

# Accuracy plot

plt.plot(np.arange(1, DIM ** 2), accuracy,
         label=label)
plt.legend()
plt.ylim((0.0, 1.1))
plt.savefig("accuracy_{}.png".format(mode))
plt.close()

# Feasibility plot
plt.plot(np.arange(1, DIM ** 2), feasibility,
         label=label)
plt.ylim((0.0, 1.1))
plt.legend()
plt.savefig("feasibility_{}.png".format(mode))
plt.close()

# Assignment frequencies plot
plt.plot(np.arange(1, DIM ** 2), tot_by_num_assigned[1:])
plt.savefig("Number of variables assigned frequencies {}.png".format(mode))
plt.close()

if args.rnd_feas:
    RANDOM_SAVE_PATH = "plots/test-pls-{}-tf-keras/random/".format(DIM)

    if args.use_prop:
        if not args.leave_columns_domains:
            RANDOM_SAVE_PATH += "rows-and-columns-prop"
        else:
            RANDOM_SAVE_PATH += "rows-prop"
    else:
        RANDOM_SAVE_PATH += "no-prop"

    try:
        os.makedirs(RANDOM_SAVE_PATH)
    except:
        print("Directory {} already exists".format(RANDOM_SAVE_PATH))

    with open("{}/random_feasibility.csv".format(RANDOM_SAVE_PATH, mode), "w") as epoch_file:
        wr = csv.writer(epoch_file)
        wr.writerow(random_feasibility)

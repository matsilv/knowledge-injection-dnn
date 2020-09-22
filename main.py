# @author: Mattia Silvestri

"""
Main program.
"""

import utility
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import csv
import argparse
import time
from models import MyModel
import math
import sys

# Set seed in order to reproduce results
#tf.random.set_seed(0)

# Tensorflow 2 GPU setup
tf.debugging.set_log_device_placement(True)
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
parser.add_argument("--load-mode" ,default="onehot", choices=["onehot", "string"],
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
parser.add_argument("--model-type", default="agnostic", choices=["agnostic", "sbrinspiredloss", "confidences"],
                    help="Choose the model type. agnostic is a simple Sequential model full agnostic. sbrinspiredloss "
                         + "envelops the SBR-inspired loss function.")
parser.add_argument("--validation-size", type=int, default=0,
                    help="Validation set dimension. If zero no validation set is used")

parser.add_argument("--use-prop", action="store_true", default=False,
                    help="True if you want to assist the with propagation method during evaluation time." +
                         "A penalty type must be assigned.")

parser.add_argument("--lmbd", default=1.0, type=float, help="Lambda for SBR-inspired term")

args = parser.parse_args()
########################################################################################################################

# Problem dimension
DIM = int(args.dim)

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

########################################################################################################################
# Create a validation set if specified

validation_set = None

if VAL_SIZE > 0:

    # Create problem instance
    val_problem = utility.PLSInstance(n=DIM, leave_columns_domains=args.leave_columns_domains)
    init_val_prob = val_problem.copy()

    # Load from test data
    load_path = "datasets/pls{}/DS.PLS.A.UNIQUES.L.4.pls{}_10k.txt".format(DIM, DIM)

    print("Loading validation data from {}".format(load_path))
    start_time = time.time()

    domains_filename = "datasets/pls{}/domains_test_10k.csv".format(DIM)

    X_val, Y_val = utility.load_dataset(load_path, val_problem, max_size=math.inf, mode=LOAD_MODE, save_domains=SAVE_DOMAINS,
                                domains_filename=domains_filename)
    end_time = time.time()
    print("Elapsed {} sec".format(end_time - start_time))

    start_time = time.time()

    if args.penalties_type == "domains":
        # Loading domains data
        print("Loading validation domains from {}".format(domains_filename))
        penalties_val = utility.read_domains_from_csv(domains_filename, max_size=X_val.shape[0])
        print()

        penalties_val = penalties_val.reshape(penalties_val.shape[0], -1)
        end_time = time.time()
        print("Elapsed {} sec".format(end_time - start_time))
    elif args.penalties_type == "multi":
        # Create forbidden multi assignments
        penalties_val = X_val.reshape((X_val.shape[0], DIM, DIM, DIM))
        penalties_val = np.sum(penalties_val, axis=3)
        penalties_val = np.repeat(penalties_val[:, :, :, np.newaxis], DIM, axis=3)
        assert penalties_val.shape == (X_val.shape[0], DIM, DIM, DIM), penalties_val.shape
        penalties_val = penalties_val.reshape(penalties_val.shape[0], -1)

    indexes = np.random.choice(np.arange(0, X_val.shape[0]), size=VAL_SIZE)
    X_val = X_val[indexes]
    if args.penalties_type is not None:
        penalties_val = penalties_val[indexes]
    else:
        penalties_val = np.zeros_like(X_val)
    validation_set = (X_val, penalties_val)

########################################################################################################################
# Load training or test data

# Create problem instance
problem = utility.PLSInstance(n=DIM, leave_columns_domains=args.leave_columns_domains)
init_prob = problem.copy()

# Load training data
load_path = "datasets/pls{}/DS.PLS.A.{}.{}.4.{}.txt".format(DIM, SOL_TYPE, mode_char, file_name)
if not TRAIN:
    load_path = "datasets/pls{}/DS.PLS.A.UNIQUES.L.4.{}.txt".format(DIM, file_name)

print("Loading data from {}".format(load_path))
start_time = time.time()
domains_filename = "datasets/pls{}/domains_{}_{}.csv".format(DIM, mode, args.num_sol)
if not TRAIN:
    domains_filename = "datasets/pls{}/domains_{}_10k.csv".format(DIM, mode)

if args.leave_columns_domains:
    domains_filename = "datasets/pls{}/rows_propagation_domains_{}_{}.csv".format(DIM, mode, args.num_sol)

X, Y = utility.load_dataset(load_path, problem, max_size=MAX_SIZE, mode=LOAD_MODE, save_domains=SAVE_DOMAINS,
                            domains_filename=domains_filename, save_partial_solutions=False,
                            partial_sols_filename="datasets/pls10/partial_sols_pls10_num_assigned_40.csv")
end_time = time.time()
print("Elapsed {} sec".format(end_time - start_time))

start_time = time.time()

if args.penalties_type == "domains":
    # Loading domains data
    print("Loading domains from {}".format(domains_filename))
    penalties = utility.read_domains_from_csv(domains_filename, max_size=X.shape[0])
    print()

    penalties = penalties.reshape(penalties.shape[0], -1)
    end_time = time.time()
    print("Elapsed {} sec".format(end_time - start_time))
elif args.penalties_type == "multi":
    # Create forbidden multi assignments
    penalties = X.reshape((X.shape[0], DIM, DIM, DIM))
    penalties = np.sum(penalties, axis=3)
    penalties = np.repeat(penalties[:, :, :, np.newaxis], DIM, axis=3)
    assert penalties.shape == (X.shape[0], DIM, DIM, DIM), penalties.shape
    penalties = penalties.reshape(penalties.shape[0], -1)

# Cast numpy array to int8 to save memory
X = X.astype(np.int8)
Y = Y.astype(np.int8)

if args.penalties_type is not None:
    penalties = penalties.astype(np.int8)
else:
    penalties = np.zeros_like(X)

# visualize pruning effect with increasing size
#X, Y, _ = utility.checker_pruning(X, Y, upper_bound=13, visualize=True)

########################################################################################################################

# Visualize data
if VISUALIZE:
    utility.visualize_dataset(X, Y, exit_end=VISUALIZE)

# Create the model
if MODEL_TYPE not in ['agnostic', 'sbrinspiredloss', 'confidences']:
    raise Exception("Model type not valid")
    exit(1)

if TRAIN:
    # Load confidence scores if the model type is confidences
    if MODEL_TYPE == 'confidences':
        if args.leave_columns_domains:
            path = "plots/test-pls-{}-validation/random/rows-prop/random_feasibility.csv".format(DIM)
        else:
            path = "plots/test-pls-{}-validation/random/rows-and-columns-prop/random_feasibility.csv".format(DIM)

        print("Loading confidences score from {}".format(path))
        
        confidences_score = np.genfromtxt('{}'.format(path), delimiter=',')
        confidences_score = np.insert(confidences_score, 0, 1.0)
        confidences = utility.from_penalties_to_confidences(X, penalties, Y, confidences_score)
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y, confidences)).shuffle(X.shape[0]).batch(batch_size=BATCH_SIZE)
        print("Training set memory size: {}".format(sys.getsizeof(train_dataset)))
        print("Confidences memory size: {}".format(sys.getsizeof(confidences)))
    else:
        #dataset = tf.data.Dataset.from_tensor_slices((X, Y, domains, multi_assign))
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y, penalties)).shuffle(X.shape[0]).batch(batch_size=BATCH_SIZE)


model = MyModel(num_layers=2, num_hidden=[512, 512], input_shape=X.shape[1:], output_dim=DIM ** 3, method=MODEL_TYPE, lmbd=args.lmbd)

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

# Train model
if TRAIN:    
    history = model.train(EPOCHS, train_dataset, "models/{}".format(model_name), DIM, validation_set, args.use_prop)
    tf.saved_model.save(model.model, "models/{}".format(model_name))
    #model.save_weights("models/{}".format(model_name), save_format='tf')

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
     # model.load_weights("models/{}".format(model_name))
     model.model = tf.saved_model.load("models/{}".format(model_name))

########################################################################################################################

# Test the model

# Make predictions
tensor_X = X.astype(np.float32)
predict_val = model.predict_from_saved_model(tensor_X).numpy()

if args.use_prop:
    predict_val *= (1 - penalties)

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
for x, pred, y, d in zip(X, predict_val, Y, penalties):

    if count % 1000 == 0:
        print("Examined {} instances".format(count))

    '''utility.visualize(x.reshape(10, 10, 10))
    print()
    utility.visualize(y.reshape(10, 10, 10))
    print()
    for i in range(DIM):
      for j in range(DIM):
        print(d.reshape(10, 10, 10)[i, j])
      print()
    print('------------------------------------------------------------------')

    exit(0)'''

    num_assigned_vars = np.sum(x.astype(np.int8))
    pred_label = np.argmax(pred.reshape(-1))
    correct_label = np.argmax(y.reshape(-1))

    #print("N. assigned features: {} | Prediction: {} | Label: {}".format(num_assigned_vars, pred_label, correct_label))

    if pred_label == correct_label:
        acc += 1
        pred_by_num_assigned[num_assigned_vars] += 1

    # Create a problem instance with current training example for net prediction
    square = np.reshape(x, (DIM, DIM, DIM))
    pls = utility.PLSInstance(n=DIM)
    pls.square = square.copy()
    # assert pls.__check_constraints__(), "Constraints should be verified before assignment"

    # Make the prediction assignment
    assignment = np.argmax(pred)
    assignment = np.unravel_index(assignment, shape=(DIM, DIM, DIM))

    # Local consistency
    local_feas = pls.assign(assignment[0], assignment[1], assignment[2])

    vals_square = np.argmax(square, axis=2) + np.sum(square, axis=2)
    solver = utility.PLSSolver(DIM, square=np.reshape(vals_square, -1))
    res = solver.solve()
    assert res, "Constraint solver is wrong because the input comes from a real solution"

    # Global consistency
    if local_feas:
        vals_square = np.argmax(pls.square.copy(), axis=2) + np.sum(pls.square.copy(), axis=2)
        solver = utility.PLSSolver(DIM, square=np.reshape(vals_square, -1), specialized=SPECIALIZED_CONSTRAINT,
                                   size=SIZE)
        feas = solver.solve()
    else:
        feas = local_feas

    if feas:
        feas_by_num_assigned[num_assigned_vars] += 1

    ####################################################################################################################

    # check random assignment performance
    '''rand_assignment = utility.random_assigner(DIM ** 3, d)
    if rand_assignment == correct_label:
        acc_rand += 1
        rand_pred_by_num_assigned[num_assigned_vars] += 1

    # create a problem instance with current training example for random prediction
    square = np.reshape(x, (DIM, DIM, DIM))
    pls = utility.PLSInstance(n=DIM)
    pls.square = square.copy()
    #assert pls.__check_constraints__(), "Constraints should be verified before assignment"

    # make the random assignment
    rand_assignment = np.unravel_index(rand_assignment, shape=(DIM, DIM, DIM))

    local_feas = pls.assign(rand_assignment[0], rand_assignment[1], rand_assignment[2])

    # global consistency
    if local_feas:
        vals_square = np.argmax(pls.square.copy(), axis=2) + np.sum(pls.square.copy(), axis=2)
        solver = utility.PLSSolver(DIM, square=np.reshape(vals_square, -1), specialized=SPECIALIZED_CONSTRAINT,
                                   size=SIZE)
        feas = solver.solve()
    else:
        feas = local_feas

    if feas:
        rand_feas_by_num_assigned[num_assigned_vars] += 1'''
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
      plt.plot(np.arange(1, DIM ** 2), feasibility,
              label=label)
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

print("Accuracy: {} | Random accuracy: {}".format(acc / count, acc_rand / count))

########################################################################################################################

# Make plots

accuracy = list((pred_by_num_assigned / (tot_by_num_assigned + 1e-8))[1:])
feasibility = list((feas_by_num_assigned / (tot_by_num_assigned + 1e-8))[1:])

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

random_feasibility = list((rand_feas_by_num_assigned / (tot_by_num_assigned + 1e-8))[1:])

# Random feasibility

plt.plot(np.arange(1, DIM ** 2), random_feasibility,
         label=label)
plt.ylim((0.0, 1.1))
plt.legend()
plt.savefig("random_feasibility.png".format(mode))

# write results on csv files
with open("{}/accuracy_{}.csv".format(SAVE_PATH, mode), "w") as epoch_file:
    wr = csv.writer(epoch_file)
    wr.writerow(accuracy)

with open("{}/feasibility_{}.csv".format(SAVE_PATH, mode), "w") as epoch_file:
    wr = csv.writer(epoch_file)
    wr.writerow(feasibility)

with open("{}/random_feasibility.csv".format(SAVE_PATH, mode), "w") as epoch_file:
    wr = csv.writer(epoch_file)
    wr.writerow(random_feasibility)

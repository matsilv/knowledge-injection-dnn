# Knowledge injection in DNN: a controlled experiment on a constrained problem

This repository contains the source code to reproduce results reported in the paper "Knowledge Injection in Deep Neural 
Networks: a Controlled Experiments on a Constrained Problem".

The followings are the steps to reproduce results. **The file names must be as indicated in the istructions below**.
The PLS-7 is chosen as demonstrating example:

1)  Create the CSV file with the solutions pool:
    `datasetgenerator/plsgen.py -o 7 -n 10000 -f bin > pls7_10k.csv`.
2)  Create the CSV file with the partial solutions - assignments pairs.  
    1) Create the file with uniques partial solutions - assignments pairs.  
    `python datasetgenerator/dataprocessing.py -n pls7_10k`  
    `DS.PLS.A.UNIQUES.B.4.pls7_10k.txt`: training set.  
    `DS.PLS.A.UNIQUES.L.4.pls7_10k.txt`: test set.  
    Then convert them to csv file and save the variables' domains after constraints propagation. For the example:
    `python dataset_to_csv.py --filename "DS.PLS.A.UNIQUES.B.4.pls7_10k.txt" 
    --partial-sols-filename "partial_solutions_10k_train.csv" 
    --domains-type full --domains-filename "domains_train_10k.csv" 
    --assignments-filename "assignments_10k_train.csv" --dim 7`  
    Do the same to save the rows constraints propagation domains:  
    `python dataset_to_csv.py --filename "DS.PLS.A.UNIQUES.B.4.pls7_10k.txt" 
    --partial-sols-filename "partial_solutions_10k_train.csv" 
    --domains-type rows --domains-filename "rows_propagation_domains_train_10k.csv" 
    --assignments-filename "assignments_10k_train.csv" --dim 7`  
    Repeat the two previous steps also for the test set:  
    2) Do the same for the multiple deconstructions of 100 solutions pool (but use the same test set achived for the 10k 
    solutions pool). 
    `python datasetgenerator/dataprocessing.py -n pls7_100.csv --sol-num 100 --iter-num 100` 
    
3) Move the files created in the previous steps in a directory named `datasets/pls7`.

4) Train and test the models.
    1. Here is an example for the model-agnostic NN:  
    `python main.py --dim 7 --train --test-num pls-7/model-agnostic/all-ts/run-1 --num-epochs 10000 --max-size 1000000 
    --batch-size 2048 --num-sol 10k --model-type agnostic --validation-size 5000 --patience 10`  
    2. Here is an example for a neural network trained with injection of the all constraints via the MSE regularization 
    method:  
    `python main.py --dim 7 --train --test-num pls-7/mse-loss/100-sols/full/run-1
    --num-epochs 10000 --max-size 1000000 --batch-size 2048 --num-sol 10k --model-type sbrinspiredloss 
    --validation-size 5000 --patience 10 --lmbd 1`  
    You can specify the `--leave-columns-domains` flag to not prune the domains according to the columns constraints.  
    3. Test model:  
    `python main.py --dim 7 --test-num pls-7/mse-loss/100-sols/full/run-1
    --num-epochs 10000 --max-size 100000 --batch-size 2048 --num-sol 10k --model-type sbrinspiredloss 
    --validation-size 0 --patience 10 --lmbd 1`  
    You can compute the random assigner feasibility adding the `--rnd`. You can assist both the loaded model and the 
    random assigner using the `--use-prop` flag.
    Test results are saved in a subdirectory of `plots`.

5) Generate the solutions starting from an empty partial solutions.  
    1. Generate `n` empty partial solutions:  
    `empty_sols = numpy.zeros(shape=(n, 7**3), dtype=numpy.int8)`  
    `numpy.savetxt('solutions/pls7/empty_sols.txt', empty_sols, delimiter=',', fmt='%0.0f')`  
    2. Generate the solutions using the trained models:  
    `cd datasetgenerator`  
    `python plstest.py ../solutions/pls7/empty_sols.csv --input-format bin --output-format bin --seed 1 
    --search-strategy snail-dnn  --max-size 5000 --dnn-fstem ../models/--test-num pls-7/model-agnostic/all-ts/run-1 
    --rm-rows-constraints --rm-columns-constraints >  ../solutions/pls7/model_agnostic_all_ts_no_prop.csv`  
    3. To count the constraints violations use the `read_solutions_from_csv` method from `utility.py`.  
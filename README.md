# Knowledge injection in DNN: a controlled experiment on a constrained problem

This repository contains the source code to reproduce results reported in the paper "Knowledge Injection in Deep Neural 
Networks: a Controlled Experiments on a Constrained Problem".

The followings are the steps to reproduce results. The PLS-7 is chosen as demonstrating example:

1)  Create the CSV file with the solutions pool:
    `datasetgenerator/plsgen.py -o 7 -n 10000 -f bin > pls7_10k.csv`.
2)  Create the CSV file with the partial solutions - assignments pairs.  
    1) Create the file with uniques partial solutions - assignments pairs.  
    `python datasetgenerator/dataprocessing.py -n pls7_10k.csv`  
    `DS.PLS.A.UNIQUES.B.4.pls7_10k.txt`: training set.  
    `DS.PLS.A.UNIQUES.L.4.pls7_10k.txt`: test set.  
    Then convert them to csv file and save the variables' domains after constraints propagation. For the example:
    `python dataset_to_csv.py --filename "DS.PLS.A.UNIQUES.B.4.pls7_10k.txt" 
    --partial-sols-filename "partial_solutions_10k_train.csv" 
    --domains-type full --domains-filename "domains_train_10k.csv" 
    --assignments-filename "assignments_10k_train.csv" --dim 7`  
    Do the same to save the rows constraints propagations domains.  
    `python dataset_to_csv.py --filename "DS.PLS.A.UNIQUES.B.4.pls7_10k.txt" 
    --partial-sols-filename "partial_solutions_10k_train.csv" 
    --domains-type rows --domains-filename "domains_train_10k.csv" 
    --assignments-filename "assignments_10k_train.csv" --dim 7`
    2) Do the same for the multiple deconstructions of 100 solutions pool.
    `python datasetgenerator/dataprocessing.py -n pls7_100.csv --sol-num 100 --iter-num 100` 
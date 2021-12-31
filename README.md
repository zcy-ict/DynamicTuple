# DynamicTuple: The Dynamic Adaptive Tuple for High-performance Packet Classification

## Introduction
The code is for the paper on Compuer Networks 2022

If there are any problems or bugs, welcome to discuss with me

zhangchunyang@ict.ac.cn

## Experimental environment

Ubuntu 16.04

g++ 5.4.0  

## Parameters
**--run_mode :**                run mode, like "classification"

**--method_name :**             the method of alogrithm

**--rules_file :**              the file of rules

**--traces_file :**             the file of traces

**--rules_shuffle :**           "1" means shuffle the rules and "0" means not

**--lookup_round :**            the lookup process repeat "n" rounds, then get the average lookup time

**--force_test :**              to verify the result of lookup, "0" means not verify, "1" means verify, "2" means verify after delete 25% rules

**--print_mode :**              the print mode, "0" means print with instructions, "1" means print without instructions

**--prefix_dims_num :**         use the combination of "k" prefix lengths to generate tuples

**--next_layer_rules_num :**    the lengths of rule chain when MultilayerTuple builds the next layer, default is "20"


## Algorithms
**PSTSS :**                     (Source Code)        the PSTSS in the source code of PartitionSort

**PartitionSort :**             (Source Code)        the source code of PartitionSort

**TupleMerge :**                (Source Code)        the source code of TupleMerge

**DimTSS :**                    (My Reproduction)    reproduction of TSS

**MultilayerTuple :**           (My Work)            reduce prefix lengths with multiple layers

**DynammicTuple_Demo :**        (My Work)            the demo to show the access numbers of each structure, need to define TUPLEINFO in dynamictuple-ranges.h

**DynammicTuple_Basic :**       (My Work)            apply 2D dynamic programming to reduce the prefix lengths of src and dst IP

**DynammicTuple :**             (My Work)            based on DynammicTuple_Basic, use the port hash table to accelerate

**DynammicTuple_Dims :**        (My Work)            based on the code of MultilayerTuple, reduce the prefix lengths for multiple dims


## Sample
sh run.sh

make && ./main --run_mode classification --method_name DynamicTuple --rules_file data/10K_acl1_rules --traces_file data/10K_acl1_traces --rules_shuffle 1 --lookup_round 10 --force_test 0 --print_mode 0 --prefix_dims_num 2

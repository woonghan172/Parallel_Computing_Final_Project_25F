# Project Description
In this project, we implement the N-Bodies Simulation.
Our separate implementations is
* Serial version
* GPU Gems version
* Thread Coarsening
* Hyperparameter tuning (change block size, coarsening factor, etc.)
* Hierarchical n-body simulation

### Group members
 - Po-Tsun Yu
 - Alexander Kachan
 - Allen Liao
 - Nicholas Drew
 - Will Han

## How to Build and Run the Code
### Build
clone this repository to a folder with a CUDA enabled GPU (via git or vscode)
then, in terminal:
./build.sh

### Run
in terminal:    
./test_one.sh [strategy] [test case num] [output file name (optional)]  
ex) ./test_one.sh 1 0 output.txt  
ex) ./test_one.sh 1 0  => default output file will be as ./results/[strategy_name]_test[test_num].txt

* strategies  
0: serial  
1: shared memory  
2: thread coarsening

## How to Run Correctness Tests
e.g., in terminal:
python3 ./run_test.py 1

* How to run a single correctness test  
./correctness_test.sh [strategy] [test case num]
ex) ./correctness_test.sh 2 0

* strategies  
0: serial  
1: shared memory  
2: thread coarsening

## How to Run Performance Tests and Reproduce Results
python3 ./run_test.py 2

## How to Run Correctness + Performance Test
python3 ./run_test.py

## Dependencies and Environment Setup

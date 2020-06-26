# DNN-MCTS for circuit routing

This is the code for Circuit Routing Using Monte Carlo Tree Search and Deep Learning. It is for running one parameter setting, which is defined in the file of "paras.txt".

## Requirements

* Python3
* Numpy
* Matplotlib
* Keras

## Running

To run the code for single test board, two files are needed: pretrained DNN model (`actor.h5`); test board (`testboard.csv`)

Then write all the parameters in the file [paras.txt](paras.txt) and run by:

```
python3 main.py paras.txt
```

To run the code for multiple test boards, revise settings in bash file `multi_run.sh` and then run by:

```
bash multi_run.sh
```
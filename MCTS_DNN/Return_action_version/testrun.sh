#!/bin/bash
n=1000
python3 paras.py $n
mkdir output_$n
# this for loop decide the number of python files to run at the same time,
# it should be adjusted according to the number of cores of the server
for i in {0..39}
do
   nohup python3 main.py ./best$n/paras_best$i.txt > output_$n/output$i.out 2>&1 &
done

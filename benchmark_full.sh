#!/bin/bash
curl -o pwd.txt http://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv && \
cd dataset/ && \
mix escript.build && \
./dataset ../pwd.txt ../benchmark_set_full ../subjects.txt && \
<<<<<<< HEAD
cd .. && \
python3 benchmark_train/benchmark.py
=======
cd .. #&& \
#python3 benchmark_train/benchmark.py
>>>>>>> Experiment on full benchmark data

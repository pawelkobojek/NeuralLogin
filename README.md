# NeuralLogin

tl;dr - User identification based merely on the way he or she uses the keyboard.

This project's purpose is solving the problem of keystroke dynamics based identification. To do so, it attempts to make use of LSTM networks.

Running the learning code (and downloading file if neeeded):
```
./run.sh
```


## Training data
Since training data contain email addresses it is not provided in the repo. For the application to work, you need to obtain `data.zip` file and run: 
```
./get_data.sh
```
Currently, doing so is exact equivalent for `unzip data.zip`, but keep in mind that this may change and former method is intended to be version-unaware (to be more specific, obviously it's better idea to perform a one way hashing of the addresses and keep them publicly available in such form, but I simply hadn't had time for that).

The email addresses used in the app are actually publicly available, but we still believe it's better not to show them aggregated in one place. This is why the mentioned `data.zip` file is currently unobtainable.

## Results
Results are simply printed to stdout as a part of the pretty verbose learning program output. Some results are also stored in `training/keras/models/$MODEL_NAME/results/` directory where $MODEL_NAME is currently one of these: lstm2layer1dropout, lstm2layer2dropout, embed2lstm. The files contain 3 lines: model prediction vector, actual class (which is binary), accuracy with threshold of 0.5. This 'data' may be used to plot ROCs.

## Plotting ROCurves
If you want to plot ROCs run `plot_roc.py` passing results directory as an argument. It will draw graph for every address in a list being part of legendary `data.zip` file.

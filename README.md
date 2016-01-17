# NeuralLogin

tl;dr - User identification based merely on the way he or she uses the keyboard.

This project's purpose is solving the problem of keystroke dynamics based identification. To do so, it attempts to make use of LSTM networks.

Running the learning code (assuming you have what's needed, see below):
```
chmod +x run.sh && ./run.sh
```


## Training data
Since training data contain email adressess it is not provided in the repo. For the application to work, you need to obtain `data.zip` file and run `chmod +x get_data.sh && ./get_data.sh`. Currently, doing so is exact equivalent for `unzip data.zip`, but keep in mind that this may change and former method is intended to be version-unaware (to be more specific, it's better idea to perform a one way hashing of the adressess and keep them publicly available, but I simply hadn't had time for thath).

The email adressess used in the app are actually publicly available, but we still believe it's better not to show them aggregated in one place. This is why the mentioned `data.zip` file is currently unobtainable.

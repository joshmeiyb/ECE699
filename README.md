# ECE699
## Downloading Code for First Time
You may need to set up an ssh certificate for git. 

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

## How to run

1. Make the file using `make`
2. Get the output using `./output`
3. If needed, clear the object files (*.o) using `make clean`

## How to run different examples
There are three different examples for you to choose:

1. Very simple integer BP: `example_fc_int_bp_very_simple()`
2. Fully connected (fc) mnist integer DFA: `example_fc_int_dfa_mnist()`
3. Fully connected (fc) fashion mnist integer DFA: `example_fc_int_dfa_fashion_mnist()`

You can comment out the current function and call the new function in the `main.cpp` file

## To be continued...
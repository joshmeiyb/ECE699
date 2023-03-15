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
2. Fully connected (fc) mnist integer BP: `example_fc_int_bp_mnist()`
3. Fully connected (fc) mnist integer DFA: `example_fc_int_dfa_mnist()`
4. Fully connected (fc) fashion mnist integer DFA: `example_fc_int_dfa_fashion_mnist()`

You can comment out the current function and call the new function in the `main.cpp` file

## TODO

1. For mnist dataset training,use classification loss function `batchCrossEntropyLoss()` and test the accuracy


## Debug
1. Error: output: pktnn_mat.cpp:566: pktnn::pktmat& pktnn::pktmat::matElemDivMat(pktnn::pktmat&, pktnn::pktmat&): Assertion `mat1.dimsEqual(mat2)' failed

    `Cause: backward() --> computeDeltas() --> matElemDivMat()`

2. Training Accuracy low, for example 9.8%
    `Cause: useDfa(false)`

3. Training Accuracy 100%, maybe not correct either
    Try to modify the layer setting to match with BP_simple_training
    
    ```
    //Modified the fc layers to do BP training
    fc1.useDfa(false).initHeWeightBias().setActv(a).setNextLayer(fc2);         //Use BP for training, set useDfa(false)
    fc2.useDfa(false).initHeWeightBias().setActv(a).setPrevLayer(fc1).setNextLayer(fcLast);
    fcLast.useDfa(false).setActv(b).setPrevLayer(fc2);
    ```
## Reference
1. Neural Networks and Deep Learning - Michael Nielsen http://neuralnetworksanddeeplearning.com/

## To be continued...
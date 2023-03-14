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

#include "pktnn_examples.h"
#include "pktnn_fc.h"

using namespace pktnn;

int example_fc_int_bp_very_simple() {
    // working setting: lrInv 1e5
    const int dim1 = 3;
    const int dim2 = 5;
    const int numEpochs = 100;

    pktmat mat1(1, dim1);
    pktfc fc1(dim1, dim2);
    pktfc fc2(dim2, 1);

    mat1.setElem(0, 0, 10);
    mat1.setElem(0, 1, 20);
    mat1.setElem(0, 2, 30);
    mat1.printMat();

    fc1.useDfa(false).initWeightBias().setActv(pktactv::Actv::pocket_tanh).setNextLayer(fc2);
    fc2.useDfa(false).initWeightBias().setActv(pktactv::Actv::as_is).setPrevLayer(fc1);

    int y = 551; // random number

    for (int i = 0; i < numEpochs; ++i) {
        fc1.forward(mat1);
        fc2.forward(fc1);

        int y_hat = fc2.mOutput.getElem(0, 0);
        int loss = pktloss::scalarL2Loss(y, y_hat);
        int lossDelta = pktloss::scalarL2LossDelta(y, y_hat);
        std::cout << "y: " << y << ", y_hat: " << y_hat << ", l2 loss: " << loss << ", l2 loss delta: " << lossDelta << "\n";

        pktmat lossDeltaMat;
        lossDeltaMat.resetZero(1, 1).setElem(0, 0, lossDelta);

        fc2.backward(lossDeltaMat, 1e5);
    }

    return 0;
}

/*
int example_fc_int_bp_mnist() {
    int numTrainSamples = 60000;
    int numTestSamples = 10000;

    pktmat mnistTrainLabels;
    pktmat mnistTrainImages;
    pktmat mnistTestLabels;
    pktmat mnistTestImages;

    pktloader::loadMnistLabels(mnistTrainLabels, numTrainSamples, true); // numTrainSamples x 1
    pktloader::loadMnistImages(mnistTrainImages, numTrainSamples, true); // numTrainSamples x (28*28)
    pktloader::loadMnistLabels(mnistTestLabels, numTestSamples, false); // numTestSamples x 1
    pktloader::loadMnistImages(mnistTestImages, numTestSamples, false); // numTestSamples x (28*28)

    // std::cout << "mnistTrainImages: ";
    // mnistTrainImages.printMat();
    // std::cout << "mnistTestImages: " << mnistTestImages.getMat() << "\n";

    std::cout << "\nLoaded train images," << numTrainSamples << ",Loaded test images," << numTestSamples << "\n";

    int numClasses = 10;
    int mnistRows = 28;
    int mnistCols = 28;

    const int dimInput = mnistRows * mnistCols;
    const int dim1 = 100;
    const int dim2 = 50;
    pktactv::Actv a = pktactv::Actv::pocket_sigmoid;
    pktactv::Actv b = pktactv::Actv::pocket_softmax;
    pktactv::Actv c = pktactv::Actv::as_is;
    pktactv::Actv d = pktactv::Actv::pocket_tanh;

    pktfc fc1(dimInput, dim1);
    pktfc fc2(dim1, dim2);
    pktfc fcLast(dim2, numClasses);
    // Modified the fc layers to do BP training
    // Use BP for training, set useDfa(false)
    fc1.useDfa(false).initWeightBias().setActv(d).setNextLayer(fc2);   
    // std::cout << "\nfc1 Weight:";
    // fc1.printWeight();
    // std::cout << "\nfc1 Bias:";
    // fc1.printBias();      
    
    fc2.useDfa(false).initWeightBias().setActv(d).setPrevLayer(fc1).setNextLayer(fcLast);
    // std::cout << "\nfc2 Weight:";
    // fc2.printWeight();
    // std::cout << "\nfc2 Bias:";
    // fc2.printBias();   
    // fc2.useDfa(false).initWeightBias().setActv(d).setNextLayer(fcLast);
    
    fcLast.useDfa(false).initWeightBias().setActv(d).setPrevLayer(fc2);
    // std::cout << "\nfcLast Weight:";
    // fcLast.printWeight();
    // std::cout << "\nfcLast Bias:";
    // fcLast.printBias();  
    // fcLast.useDfa(false).initWeightBias().setActv(d);
    
    // std::cout << "\nfc1 weight:";
    // fc1.printWeight();
    // std::cout << "\nfc1 bias:";
    // fc1.printBias();
    // std::cout << "\nfc2 weight:";
    // fc2.printWeight();
    // std::cout << "\nfc2 bias:";
    // fc2.printBias();
    // std::cout << "\nfcLast weight:";
    // fcLast.printWeight();
    // std::cout << "\nfcLast bias:";
    // fcLast.printBias();


    // initialization
    pktmat trainTargetMat(numTrainSamples, numClasses);
    pktmat testTargetMat(numTestSamples, numClasses);

    int numCorrect = 0;
    fc1.forward(mnistTrainImages);
    for (int r = 0; r < numTrainSamples; ++r) {
        trainTargetMat.setElem(r, mnistTrainLabels.getElem(r, 0), UNSIGNED_4BIT_MAX);
        if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Initial training numCorrect," << numCorrect << "\n";

    numCorrect = 0;
    fc1.forward(mnistTestImages);
    for (int r = 0; r < numTestSamples; ++r) {
        testTargetMat.setElem(r, mnistTestLabels.getElem(r, 0), UNSIGNED_4BIT_MAX);
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Initial test numCorrect," << numCorrect << "\n";
    std::cout << "----- NOW START -----\n(CSV format)\n";

    pktmat lossMat;
    pktmat lossDeltaMat;
    //pktmat batchLossDeltaMat;
    pktmat miniBatchImages;
    pktmat miniBatchTrainTargets;

    int numEpochs = 30;
    int miniBatchSize = 20; // CAUTION: Too big minibatch size can cause overflow
    int lrInv = 1000;        // Learning Rate

    std::cout << "Learning Rate Inverse," << lrInv <<
        ",numTrainSamples," << numTrainSamples <<
        ",miniBatchSize," << miniBatchSize << "\n";

    // random indices template
    int* indices = new int[numTrainSamples];
    for (int i = 0; i < numTrainSamples; ++i) {
        indices[i] = i;
    }

    std::string testCorrect = "";
    std::cout << "Training\nEpoch,SumLoss,NumCorrect,Accuracy\n";

    for (int e = 0; e < numEpochs; e++) {
        // Shuffle the indices
        for (int i = numTrainSamples - 1; i > 0; --i) {
            int j = rand() % (i + 1); // Pick a random index from 0 to r
            int temp = indices[j];
            indices[j] = indices[i];
            indices[i] = temp;
        }

        if ((e % 10 == 0) && (lrInv < 2 * lrInv)) {
            // reducing the learning rate by a half every 5 epochs
            // avoid overflow
            lrInv *= 2;
        }

        int sumLoss = 0;
        int sumLossDelta = 0;
        int epochNumCorrect = 0;
        int numIter = numTrainSamples / miniBatchSize;

        for (int i = 0; i < numIter; ++i) {
            int batchNumCorrect = 0;
            const int idxStart = i * miniBatchSize;
            const int idxEnd = idxStart + miniBatchSize;

            // std::cout << "\nmnistTrainImages: ";
            // mnistTrainImages.printMat();
            // std::cout << "\nindices: " << indices <<  "\nidxStart: " << idxStart << "\nidxEnd: " << idxEnd;
            miniBatchImages.indexedSlicedSamplesOf(mnistTrainImages, indices, idxStart, idxEnd);
            //For debug only
            //std::cout << "For miniBatchImages, indices:" << indices << "idxStart:" << idxStart << "idxEnd:" << idxEnd << "\n";              
            miniBatchTrainTargets.indexedSlicedSamplesOf(trainTargetMat, indices, idxStart, idxEnd);
            //For debug only
            //std::cout << "For miniBatchTrainTargets, indices:" << indices << "idxStart:" << idxStart << "idxEnd:" << idxEnd << "\n";        

            std::cout << "\nminiBatchImages: ";
            miniBatchImages.printMat();

            fc1.forward(miniBatchImages);
            std::cout << "\nfc1 output:";
            fc1.printOutput();
            // std::cout << "\nfc2 output:";
            // fc2.printOutput();
            // std::cout << "\nfcLast output:";
            // fcLast.printOutput();
            // fc2.forward(fc1);
            // fcLast.forward(fc2);
            sumLoss += pktloss::batchPocketCrossLoss(lossMat, miniBatchTrainTargets, fcLast.mOutput);
            // sumLoss += pktloss::batchCrossEntropyLoss(lossMat, miniBatchTrainTargets, fcLast.mOutput);
            //For debug only
            //std::cout << "sumLoss increases, sumLoss = " << sumLoss << "\n"; 
            sumLossDelta = pktloss::batchPocketCrossLossDelta(lossDeltaMat, miniBatchTrainTargets, fcLast.mOutput);
            // sumLossDelta = pktloss::batchCrossEntropyLossDelta(lossDeltaMat, miniBatchTrainTargets, fcLast.mOutput);
            //For debug only, get the gradient of sumLoss
            //std::cout << "sumLossDelta is calculated, sumLossDelta = " << sumLossDelta << "\n";

            for (int r = 0; r < miniBatchSize; ++r) {
                if (miniBatchTrainTargets.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
                    ++batchNumCorrect;
                    //For debug only
                    // std::cout << "batchNumCorrect = " << batchNumCorrect << "\n";
                }
            }
            //lossDeltaMat.resetZero(1, 1).setElem(0, 0, sumLossDelta);
            fcLast.backward(lossDeltaMat, lrInv);       //Training
            epochNumCorrect += batchNumCorrect;
            //For debug only
            //std::cout << "epochNumCorrect = " << epochNumCorrect << "\n";
        }
        std::cout << e << "," << sumLoss << "," << epochNumCorrect << "," << (epochNumCorrect * 1.0 / numTrainSamples) << "\n";

        // check the test set accuracy
        fc1.forward(mnistTestImages);
        int testNumCorrect = 0;
        for (int r = 0; r < numTestSamples; ++r) {
            if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
                ++testNumCorrect;
                //For debug only
                //std::cout << "testNumCorrect = " << testNumCorrect << "\n";
            }
        }
        testCorrect += (std::to_string(e) + "," + std::to_string(testNumCorrect) + "\n");

        // std::cout << "\nfc1 weight:";
        // fc1.printWeight();
        // std::cout << "\nfc1 bias:";
        // fc1.printBias();
        // std::cout << "\nfc2 weight:";
        // fc2.printWeight();
        // std::cout << "\nfc2 bias:";
        // fc2.printBias();
        // std::cout << "\nfcLast weight:";
        // fcLast.printWeight();
        // std::cout << "\nfcLast bias:";
        // fcLast.printBias();
    }

    fc1.forward(mnistTrainImages);
    numCorrect = 0;
    for (int r = 0; r < numTrainSamples; ++r) {
        if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
            //For debug only
            //std::cout << "For mnist train image, numCorrect = " << numCorrect << "\n";
        }
    }
    std::cout << "Final training numCorrect," << numCorrect << "\n";

    fc1.forward(mnistTestImages);
    numCorrect = 0;
    for (int r = 0; r < numTestSamples; ++r) {
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
            //For debug only
            //std::cout << "For mnist test image, numCorrect = " << numCorrect << "\n";
        }
    }
    std::cout << "\nTest\nEpoch,NumCorrect\n";
    std::cout << testCorrect;
    std::cout << "Final test numCorrect," << numCorrect << "\n";
    std::cout << "Final test accuracy," << (numCorrect * 1.0 / numTestSamples) << "\n";
    std::cout << "Final learning rate inverse," << lrInv << "\n";

    delete[] indices;

    
    return 0;
}
*/

int example_fc_int_bp_mnist() {
    int numTrainSamples = 60000;
    int numTestSamples = 10000;
    // int numTrainSamples = 1;
    // int numTestSamples = 1;

    pktmat mnistTrainLabels;
    pktmat mnistTrainImages;
    pktmat mnistTestLabels;
    pktmat mnistTestImages;

    pktloader::loadMnistLabels(mnistTrainLabels, numTrainSamples, true); // numTrainSamples x 1
    pktloader::loadMnistImages(mnistTrainImages, numTrainSamples, true); // numTrainSamples x (28*28)
    pktloader::loadMnistLabels(mnistTestLabels, numTestSamples, false); // numTestSamples x 1
    pktloader::loadMnistImages(mnistTestImages, numTestSamples, false); // numTestSamples x (28*28)
    // pktloader::loadMnistLabels(mnistTrainLabels, 1, true); // numTrainSamples x 1
    // pktloader::loadMnistImages(mnistTrainImages, 1, true); // numTrainSamples x (28*28)
    // pktloader::loadMnistLabels(mnistTestLabels, 1, false); // numTestSamples x 1
    // pktloader::loadMnistImages(mnistTestImages, 1, false); // numTestSamples x (28*28)

    std::cout << "Loaded train images," << numTrainSamples << ",Loaded test images," << numTestSamples << "\n";

    int numClasses = 10;
    int mnistRows = 28;
    int mnistCols = 28;

    const int dimInput = mnistRows * mnistCols;
    const int dim1 = 100;
    const int dim2 = 50;
    pktactv::Actv a = pktactv::Actv::pocket_tanh;
    pktactv::Actv b = pktactv::Actv::as_is;

    pktfc fc1(dimInput, dim1);
    pktfc fc2(dim1, dim2);
    pktfc fcLast(dim2, numClasses);
    fc1.useDfa(false).initWeightBias().setActv(a).setNextLayer(fc2);
    fc2.useDfa(false).initWeightBias().setActv(a).setNextLayer(fcLast);
    fcLast.useDfa(false).initWeightBias().setActv(b);
    
    // initialization
    pktmat trainTargetMat(numTrainSamples, numClasses);
    pktmat testTargetMat(numTestSamples, numClasses);

    int numCorrect = 0;
    fc1.forward(mnistTrainImages);
    // std::cout << "\nfc1 output after forwarding mnistTrainImages:";
    // fc1.printOutput();
    for (int r = 0; r < numTrainSamples; ++r) {
        trainTargetMat.setElem(r, mnistTrainLabels.getElem(r, 0), UNSIGNED_4BIT_MAX);
        if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Initial training numCorrect," << numCorrect << "\n";

    numCorrect = 0;
    fc1.forward(mnistTestImages);
    // std::cout << "\nfc1 output after forwarding mnistTestImages:";
    // fc1.printOutput();
    for (int r = 0; r < numTestSamples; ++r) {
        testTargetMat.setElem(r, mnistTestLabels.getElem(r, 0), UNSIGNED_4BIT_MAX);
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Initial test numCorrect," << numCorrect << "\n";
    std::cout << "----- NOW START -----\n(CSV format)\n";

    pktmat lossMat;
    pktmat lossDeltaMat;
    pktmat batchLossDeltaMat;
    pktmat miniBatchImages;
    pktmat miniBatchTrainTargets;

    int epoch = 3;
    int miniBatchSize = 20; // CAUTION: Too big minibatch size can cause overflow
    int lrInv = 1000;

    std::cout << "Learning Rate Inverse," << lrInv <<
        ",numTrainSamples," << numTrainSamples <<
        ",miniBatchSize," << miniBatchSize << "\n";

    // random indices template
    int* indices = new int[numTrainSamples];
    for (int i = 0; i < numTrainSamples; ++i) {
        indices[i] = i;
    }

    std::string testCorrect = "";
    std::cout << "Training\nEpoch,SumLoss,NumCorrect,Accuracy\n";
    for (int e = 1; e <= epoch; ++e) {
        // Shuffle the indices
        for (int i = numTrainSamples - 1; i > 0; --i) {
            int j = rand() % (i + 1); // Pick a random index from 0 to r
            int temp = indices[j];
            indices[j] = indices[i];
            indices[i] = temp;
        }

        if ((e % 10 == 0) && (lrInv < 2 * lrInv)) {
            // reducing the learning rate by a half every 5 epochs
            // avoid overflow
            lrInv *= 2;
        }

        int sumLoss = 0;
        int sumLossDelta = 0;
        int epochNumCorrect = 0;
        int numIter = numTrainSamples / miniBatchSize;

        for (int i = 0; i < numIter; ++i) {
            int batchNumCorrect = 0;
            const int idxStart = i * miniBatchSize;
            const int idxEnd = idxStart + miniBatchSize;
            miniBatchImages.indexedSlicedSamplesOf(mnistTrainImages, indices, idxStart, idxEnd);
            miniBatchTrainTargets.indexedSlicedSamplesOf(trainTargetMat, indices, idxStart, idxEnd);
            
            // std::cout << "\nminiBatchImages: ";
            // miniBatchImages.printMat();
            fc1.forward(miniBatchImages);
            // std::cout << "\nfc1 output:";
            // fc1.printOutput();

            sumLoss += pktloss::batchPocketCrossLoss(lossMat, miniBatchTrainTargets, fcLast.mOutput);
            sumLossDelta = pktloss::batchPocketCrossLossDelta(lossDeltaMat, miniBatchTrainTargets, fcLast.mOutput);

            for (int r = 0; r < miniBatchSize; ++r) {
                if (miniBatchTrainTargets.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
                    ++batchNumCorrect;
                }
            }
            fcLast.backward(lossDeltaMat, lrInv);
            epochNumCorrect += batchNumCorrect;
        }
        // std::cout << "Single epoch is done, check check the fc1 layer output and weights\n";
        // fc1.printOutput();
        // // fc1.();
        // fc1.printWeight();

        std::cout << e << "," << sumLoss << "," << epochNumCorrect << "," << (epochNumCorrect * 1.0 / numTrainSamples) << "\n";

        // check the test set accuracy
        fc1.forward(mnistTestImages);
        int testNumCorrect = 0;
        for (int r = 0; r < numTestSamples; ++r) {
            if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
                ++testNumCorrect;
            }
        }
        testCorrect += (std::to_string(e) + "," + std::to_string(testNumCorrect) + "\n");
    }

    fc1.forward(mnistTrainImages);
    numCorrect = 0;
    for (int r = 0; r < numTrainSamples; ++r) {
        if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Final training numCorrect," << numCorrect << "\n";

    fc1.forward(mnistTestImages);
    numCorrect = 0;
    for (int r = 0; r < numTestSamples; ++r) {
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "\nTest\nEpoch,NumCorrect\n";
    std::cout << testCorrect;
    std::cout << "Final test numCorrect," << numCorrect << "\n";
    std::cout << "Final test accuracy," << (numCorrect * 1.0 / numTestSamples) << "\n";
    std::cout << "Final learning rate inverse," << lrInv << "\n";

    delete[] indices;
    return 0;
}

int example_fc_int_dfa_mnist() {
    int numTrainSamples = 60000;
    int numTestSamples = 10000;

    pktmat mnistTrainLabels;
    pktmat mnistTrainImages;
    pktmat mnistTestLabels;
    pktmat mnistTestImages;

    pktloader::loadMnistLabels(mnistTrainLabels, numTrainSamples, true); // numTrainSamples x 1
    pktloader::loadMnistImages(mnistTrainImages, numTrainSamples, true); // numTrainSamples x (28*28)
    pktloader::loadMnistLabels(mnistTestLabels, numTestSamples, false); // numTestSamples x 1
    pktloader::loadMnistImages(mnistTestImages, numTestSamples, false); // numTestSamples x (28*28)

    std::cout << "Loaded train images," << numTrainSamples << ",Loaded test images," << numTestSamples << "\n";

    int numClasses = 10;
    int mnistRows = 28;
    int mnistCols = 28;

    const int dimInput = mnistRows * mnistCols;
    const int dim1 = 100;
    const int dim2 = 50;
    pktactv::Actv a = pktactv::Actv::pocket_tanh;

    pktfc fc1(dimInput, dim1);
    pktfc fc2(dim1, dim2);
    pktfc fcLast(dim2, numClasses);
    fc1.useDfa(true).setActv(a).setNextLayer(fc2);
    fc2.useDfa(true).setActv(a).setNextLayer(fcLast);
    fcLast.useDfa(true).setActv(a);
    
    // initialization
    pktmat trainTargetMat(numTrainSamples, numClasses);
    pktmat testTargetMat(numTestSamples, numClasses);

    int numCorrect = 0;
    fc1.forward(mnistTrainImages);
    // std::cout << "\nfc1 output after forwarding mnistTrainImages:";
    // fc1.printOutput();
    for (int r = 0; r < numTrainSamples; ++r) {
        trainTargetMat.setElem(r, mnistTrainLabels.getElem(r, 0), UNSIGNED_4BIT_MAX);
        if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Initial training numCorrect," << numCorrect << "\n";

    numCorrect = 0;
    fc1.forward(mnistTestImages);
    // std::cout << "\nfc1 output after forwarding mnistTestImages:";
    // fc1.printOutput();
    for (int r = 0; r < numTestSamples; ++r) {
        testTargetMat.setElem(r, mnistTestLabels.getElem(r, 0), UNSIGNED_4BIT_MAX);
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Initial test numCorrect," << numCorrect << "\n";
    std::cout << "----- NOW START -----\n(CSV format)\n";

    pktmat lossMat;
    pktmat lossDeltaMat;
    pktmat batchLossDeltaMat;
    pktmat miniBatchImages;
    pktmat miniBatchTrainTargets;

    int epoch = 3;
    int miniBatchSize = 20; // CAUTION: Too big minibatch size can cause overflow
    int lrInv = 1000;

    std::cout << "Learning Rate Inverse," << lrInv <<
        ",numTrainSamples," << numTrainSamples <<
        ",miniBatchSize," << miniBatchSize << "\n";

    // random indices template
    int* indices = new int[numTrainSamples];
    for (int i = 0; i < numTrainSamples; ++i) {
        indices[i] = i;
    }

    std::string testCorrect = "";
    std::cout << "Training\nEpoch,SumLoss,NumCorrect,Accuracy\n";
    for (int e = 1; e <= epoch; ++e) {
        // Shuffle the indices
        for (int i = numTrainSamples - 1; i > 0; --i) {
            int j = rand() % (i + 1); // Pick a random index from 0 to r
            int temp = indices[j];
            indices[j] = indices[i];
            indices[i] = temp;
        }

        if ((e % 10 == 0) && (lrInv < 2 * lrInv)) {
            // reducing the learning rate by a half every 5 epochs
            // avoid overflow
            lrInv *= 2;
        }

        int sumLoss = 0;
        int sumLossDelta = 0;
        int epochNumCorrect = 0;
        int numIter = numTrainSamples / miniBatchSize;

        for (int i = 0; i < numIter; ++i) {
            int batchNumCorrect = 0;
            const int idxStart = i * miniBatchSize;
            const int idxEnd = idxStart + miniBatchSize;
            miniBatchImages.indexedSlicedSamplesOf(mnistTrainImages, indices, idxStart, idxEnd);
            miniBatchTrainTargets.indexedSlicedSamplesOf(trainTargetMat, indices, idxStart, idxEnd);
            
            // std::cout << "\nminiBatchImages: ";
            // miniBatchImages.printMat();
            fc1.forward(miniBatchImages);
            // std::cout << "\nfc1 output:";
            // fc1.printOutput();

            sumLoss += pktloss::batchL2Loss(lossMat, miniBatchTrainTargets, fcLast.mOutput);
            sumLossDelta = pktloss::batchL2LossDelta(lossDeltaMat, miniBatchTrainTargets, fcLast.mOutput);

            for (int r = 0; r < miniBatchSize; ++r) {
                if (miniBatchTrainTargets.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
                    ++batchNumCorrect;
                }
            }
            fcLast.backward(lossDeltaMat, lrInv);
            epochNumCorrect += batchNumCorrect;
        }
        std::cout << e << "," << sumLoss << "," << epochNumCorrect << "," << (epochNumCorrect * 1.0 / numTrainSamples) << "\n";

        // check the test set accuracy
        fc1.forward(mnistTestImages);
        int testNumCorrect = 0;
        for (int r = 0; r < numTestSamples; ++r) {
            if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
                ++testNumCorrect;
            }
        }
        testCorrect += (std::to_string(e) + "," + std::to_string(testNumCorrect) + "\n");
    }

    fc1.forward(mnistTrainImages);
    numCorrect = 0;
    for (int r = 0; r < numTrainSamples; ++r) {
        if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Final training numCorrect," << numCorrect << "\n";

    fc1.forward(mnistTestImages);
    numCorrect = 0;
    for (int r = 0; r < numTestSamples; ++r) {
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "\nTest\nEpoch,NumCorrect\n";
    std::cout << testCorrect;
    std::cout << "Final test numCorrect," << numCorrect << "\n";
    std::cout << "Final test accuracy," << (numCorrect * 1.0 / numTestSamples) << "\n";
    std::cout << "Final learning rate inverse," << lrInv << "\n";

    delete[] indices;
    return 0;
}

int example_fc_int_dfa_fashion_mnist() {
    int numTrainSamples = 60000;
    int numTestSamples = 10000;

    pktmat mnistTrainLabels;
    pktmat mnistTrainImages;
    pktmat mnistTestLabels;
    pktmat mnistTestImages;

    pktloader::loadFashionMnistLabels(mnistTrainLabels, numTrainSamples, true); // numTrainSamples x 1
    pktloader::loadFashionMnistImages(mnistTrainImages, numTrainSamples, true); // numTrainSamples x (28*28)
    pktloader::loadFashionMnistLabels(mnistTestLabels, numTestSamples, false); // numTestSamples x 1
    pktloader::loadFashionMnistImages(mnistTestImages, numTestSamples, false); // numTestSamples x (28*28)

    std::cout << "Loaded train images," << numTrainSamples << ",Loaded test images," << numTestSamples << "\n";

    int numClasses = 10;
    int mnistRows = 28;
    int mnistCols = 28;

    const int dimInput = mnistRows * mnistCols;
    const int dim1 = 200;
    const int dim2 = 100;
    const int dim3 = 50;
    pktactv::Actv a = pktactv::Actv::pocket_tanh;

    pktfc fc1(dimInput, dim1);
    pktfc fc2(dim1, dim2);
    pktfc fc3(dim2, dim3);
    pktfc fcLast(dim3, numClasses);
    fc1.useDfa(true).setActv(a).setNextLayer(fc2);
    fc2.useDfa(true).setActv(a).setNextLayer(fc3);
    fc3.useDfa(true).setActv(a).setNextLayer(fcLast);
    fcLast.useDfa(true).setActv(a);

    // initialization
    pktmat trainTargetMat(numTrainSamples, numClasses);
    pktmat testTargetMat(numTestSamples, numClasses);

    int numCorrect = 0;
    fc1.forward(mnistTrainImages);
    for (int r = 0; r < numTrainSamples; ++r) {
        trainTargetMat.setElem(r, mnistTrainLabels.getElem(r, 0), UNSIGNED_4BIT_MAX);
        if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Initial training numCorrect," << numCorrect << "\n";

    numCorrect = 0;
    fc1.forward(mnistTestImages);
    for (int r = 0; r < numTestSamples; ++r) {
        testTargetMat.setElem(r, mnistTestLabels.getElem(r, 0), UNSIGNED_4BIT_MAX);
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Initial test numCorrect," << numCorrect << "\n";
    std::cout << "----- NOW START -----\n(CSV format)\n";

    pktmat lossMat;
    pktmat lossDeltaMat;
    pktmat batchLossDeltaMat;
    pktmat miniBatchImages;
    pktmat miniBatchTrainTargets;

    int epoch = 3;
    int miniBatchSize = 20; // CAUTION: Too big minibatch size can cause overflow
    int lrInv = 1000;

    std::cout << "Learning Rate Inverse," << lrInv <<
        ",numTrainSamples," << numTrainSamples <<
        ",miniBatchSize," << miniBatchSize << "\n";

    // random indices template
    int* indices = new int[numTrainSamples];
    for (int i = 0; i < numTrainSamples; ++i) {
        indices[i] = i;
    }

    std::string testCorrect = "";
    std::cout << "Training\nEpoch,SumLoss,NumCorrect,Accuracy\n";
    for (int e = 1; e <= epoch; ++e) {
        // Shuffle the indices
        for (int i = numTrainSamples - 1; i > 0; --i) {
            int j = rand() % (i + 1); // Pick a random index from 0 to r
            int temp = indices[j];
            indices[j] = indices[i];
            indices[i] = temp;
        }

        if ((e % 10 == 0) && (lrInv < 2 * lrInv)) {
            // reducing the learning rate by a half every 5 epochs
            // avoid overflow
            lrInv *= 2;
        }

        int sumLoss = 0;
        int sumLossDelta = 0;
        int epochNumCorrect = 0;
        int numIter = numTrainSamples / miniBatchSize;

        for (int i = 0; i < numIter; ++i) {
            int batchNumCorrect = 0;
            const int idxStart = i * miniBatchSize;
            const int idxEnd = idxStart + miniBatchSize;
            miniBatchImages.indexedSlicedSamplesOf(mnistTrainImages, indices, idxStart, idxEnd);
            miniBatchTrainTargets.indexedSlicedSamplesOf(trainTargetMat, indices, idxStart, idxEnd);

            fc1.forward(miniBatchImages);
            sumLoss += pktloss::batchL2Loss(lossMat, miniBatchTrainTargets, fcLast.mOutput);
            sumLossDelta = pktloss::batchL2LossDelta(lossDeltaMat, miniBatchTrainTargets, fcLast.mOutput);

            for (int r = 0; r < miniBatchSize; ++r) {
                if (miniBatchTrainTargets.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
                    ++batchNumCorrect;
                }
            }
            fcLast.backward(lossDeltaMat, lrInv);
            epochNumCorrect += batchNumCorrect;
        }
        std::cout << e << "," << sumLoss << "," << epochNumCorrect << "," << (epochNumCorrect * 1.0 / numTrainSamples) << "\n";

        // check the test set accuracy
        fc1.forward(mnistTestImages);
        int testNumCorrect = 0;
        for (int r = 0; r < numTestSamples; ++r) {
            if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
                ++testNumCorrect;
            }
        }
        testCorrect += (std::to_string(e) + "," + std::to_string(testNumCorrect) + "\n");
    }

    fc1.forward(mnistTrainImages);
    numCorrect = 0;
    for (int r = 0; r < numTrainSamples; ++r) {
        if (trainTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "Final training numCorrect," << numCorrect << "\n";

    fc1.forward(mnistTestImages);
    numCorrect = 0;
    for (int r = 0; r < numTestSamples; ++r) {
        if (testTargetMat.getMaxIndexInRow(r) == fcLast.mOutput.getMaxIndexInRow((r))) {
            ++numCorrect;
        }
    }
    std::cout << "\nTest\nEpoch,NumCorrect\n";
    std::cout << testCorrect;
    std::cout << "Final test numCorrect," << numCorrect << "\n";
    std::cout << "Final test accuracy," << (numCorrect * 1.0 / numTestSamples) << "\n";
    std::cout << "Final learning rate inverse," << lrInv << "\n";

    delete[] indices;
    return 0;
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model;

import Utils.FileIO;
import Utils.Utils;
import java.util.ArrayList;
import java.util.List;

/**
 * 这是一个普通的LSTM
 *
 * @author gzzengzihang
 */
public class LSTM {

    private static final String modelName = "LSTM";
    private int nodeNum;
    private int inputNum;
    private int hiddenNum;
    private int outputNum;
    private double learningRate;
    private boolean ifSequence;
    private Activation sigmoidActivation;
    private Activation tanhActivation;

    // matrix
    private double[][] forgetInputMatrix;
    private double[][] forgetHiddenMatrix;
    private double[][] inputInputMatrix;
    private double[][] inputHiddenMatrix;
    private double[][] ceremoryInputMatrix;
    private double[][] ceremoryHiddenMatrix;
    private double[][] outputInputMatrix;
    private double[][] outputHiddenMatrix;
    private double[][] outputMatrix;

    // update matrix
    private double[][] updateForgetInputMatrix;
    private double[][] updateForgetHiddenMatrix;
    private double[][] updateInputInputMatrix;
    private double[][] updateInputHiddenMatrix;
    private double[][] updateCeremoryInputMatrix;
    private double[][] updateCeremoryHiddenMatrix;
    private double[][] updateOutputInputMatrix;
    private double[][] updateOutptuHiddenMatrix;
    private double[][] updateOutputMatrix;

    // hidden values
    private List<double[][]> outputHiddenValues;
    private List<double[][]> ceremoryHiddenValues;
    private List<double[][]> outputs;

    // loss function ==> 1 / 2 * (y - d)^2
    public LSTM(int nodeNum, int inputNum, int hiddenNum, int outputNum,
            double learningRate, boolean ifSequence) {
        this.nodeNum = nodeNum;
        this.inputNum = inputNum;
        this.hiddenNum = hiddenNum;
        this.outputNum = outputNum;
        this.ifSequence = ifSequence;
        this.learningRate = learningRate;

        this.sigmoidActivation = new Activation("sigmoid");
        this.tanhActivation = new Activation("tanh");

        this.forgetInputMatrix = Utils.randomMatrix(inputNum, hiddenNum);
        this.inputInputMatrix = Utils.randomMatrix(inputNum, hiddenNum);
        this.ceremoryInputMatrix = Utils.randomMatrix(inputNum, hiddenNum);
        this.outputInputMatrix = Utils.randomMatrix(inputNum, hiddenNum);
        this.forgetHiddenMatrix = Utils.randomMatrix(hiddenNum, hiddenNum);
        this.inputHiddenMatrix = Utils.randomMatrix(hiddenNum, hiddenNum);
        this.ceremoryHiddenMatrix = Utils.randomMatrix(hiddenNum, hiddenNum);
        this.outputHiddenMatrix = Utils.randomMatrix(hiddenNum, hiddenNum);
        this.outputMatrix = Utils.randomMatrix(hiddenNum, outputNum);

        this.updateForgetInputMatrix = new double[inputNum][hiddenNum];
        this.updateInputInputMatrix = new double[inputNum][hiddenNum];
        this.updateCeremoryInputMatrix = new double[inputNum][hiddenNum];
        this.updateOutputInputMatrix = new double[inputNum][hiddenNum];
        this.updateForgetHiddenMatrix = new double[hiddenNum][hiddenNum];
        this.updateInputHiddenMatrix = new double[hiddenNum][hiddenNum];
        this.updateCeremoryHiddenMatrix = new double[hiddenNum][hiddenNum];
        this.updateOutptuHiddenMatrix = new double[hiddenNum][hiddenNum];
        this.updateOutputMatrix = new double[hiddenNum][outputNum];

        this.outputHiddenValues = new ArrayList<>();
        this.ceremoryHiddenValues = new ArrayList<>();
        this.outputs = new ArrayList<>();
    }

    public void forwardCompute(List<Data> dataList, boolean ifOutput) {
        this.outputHiddenValues.add(new double[1][hiddenNum]);
        this.ceremoryHiddenValues.add(new double[1][hiddenNum]);
        for (int i = 0; i < nodeNum; i++) {
            Data data = dataList.get(i);
            double[][] prevOutputHiddenValue = outputHiddenValues.get(i);
            double[][] prevCeremoryHiddenValue = ceremoryHiddenValues.get(i);
            double[][] forgetValue = sigmoidActivation.activation(Utils.add(
                    Utils.dot(data.getDataMatrix(), forgetInputMatrix),
                    Utils.dot(prevOutputHiddenValue, forgetHiddenMatrix)));
            double[][] inputValue = sigmoidActivation.activation(Utils.add(
                    Utils.dot(data.getDataMatrix(), inputInputMatrix),
                    Utils.dot(prevOutputHiddenValue, inputHiddenMatrix)));
            double[][] ceremoryValue = tanhActivation.activation(Utils.add(
                    Utils.dot(data.getDataMatrix(), ceremoryInputMatrix),
                    Utils.dot(prevOutputHiddenValue, ceremoryHiddenMatrix)));
            ceremoryValue = Utils.add(Utils.multiple(forgetValue, prevCeremoryHiddenValue),
                    Utils.multiple(inputValue, ceremoryValue));
            this.ceremoryHiddenValues.add(ceremoryValue);
            double[][] outputHiddenValue = sigmoidActivation.activation(Utils.add(
                            Utils.dot(data.getDataMatrix(), outputInputMatrix),
                            Utils.dot(prevOutputHiddenValue, outputHiddenMatrix)));
                    outputHiddenValue = Utils.multiple(outputHiddenValue, tanhActivation.activation(ceremoryValue));
            this.outputHiddenValues.add(outputHiddenValue);
            double[][] output = new double[1][1];
            if (!ifSequence){
                if (i == nodeNum - 1){
                    output = sigmoidActivation.activation(Utils.dot(outputHiddenValue, outputMatrix));
                    this.outputs.add(output);
                }
            }else{
                output = sigmoidActivation.activation(Utils.dot(outputHiddenValue, outputMatrix));
                this.outputs.add(output);
            }

        }
        if (ifOutput) {
            for (double[][] output : outputs) {
                System.out.println(output[0][0]);
            }
        }
    }

    private void backwardCompute(List<Data> dataList) {
        List<double[][]> futureOutputHiddenDeltas = new ArrayList<>();
        List<double[][]> futureCeremoryHiddenDeltas = new ArrayList<>();
        futureOutputHiddenDeltas.add(new double[1][hiddenNum]);
        futureCeremoryHiddenDeltas.add(new double[1][hiddenNum]);
        int futureCount = 0;
        for (int i = nodeNum - 1; i >= 0; i--) {
            Data data = dataList.get(i);
            double[][] outputHiddenValue = outputHiddenValues.get(i + 1);
            double[][] prevoutputHiddenValue = outputHiddenValues.get(i);
            double[][] ceremoryHiddenValue = ceremoryHiddenValues.get(i + 1);
            double[][] prevCeremoryHiddenValue = ceremoryHiddenValues.get(i);
            double[][] futureOutputHiddenDelta = futureOutputHiddenDeltas.get(futureCount);
            double[][] futureCeremoryHiddenDelta = futureCeremoryHiddenDeltas.get(futureCount);
            double[][] outputMatrixDelta;
            
            if (!ifSequence){
                if (i == nodeNum - 1){
                    double[][] error = Utils.sub(data.getTarget(), outputs.get(0));
                    // output x output matrix
                    outputMatrixDelta = Utils.multiple(Utils.dot(error, -1),
                            sigmoidActivation.activationDerivative(Utils.dot(outputHiddenValue, outputMatrix)));
                    this.updateOutputMatrix = Utils.add(this.updateOutputMatrix,
                            Utils.dot(Utils.transposition(outputHiddenValue), outputMatrixDelta));
                }else{
                    outputMatrixDelta = new double[1][1];
                }
            }else{
                double[][] error = Utils.sub(data.getTarget(), outputs.get(i));
                // output x output matrix
                outputMatrixDelta = Utils.multiple(Utils.dot(error, -1),
                        sigmoidActivation.activationDerivative(Utils.dot(outputHiddenValue, outputMatrix)));
                this.updateOutputMatrix = Utils.add(this.updateOutputMatrix,
                        Utils.dot(Utils.transposition(outputHiddenValue), outputMatrixDelta));
            }

            // output x hidden matrix
            double[][] outputIHMatrixDelta = Utils.multiple(Utils.multiple(Utils.add(Utils.dot(outputMatrixDelta,
                    Utils.transposition(outputMatrix)), futureOutputHiddenDelta),
                    tanhActivation.activation(ceremoryHiddenValue)), sigmoidActivation.
                    activationDerivative(Utils.add(Utils.dot(data.getDataMatrix(),
                            outputInputMatrix), Utils.dot(prevoutputHiddenValue, outputHiddenMatrix))));

            futureOutputHiddenDeltas.add(outputIHMatrixDelta);

            this.updateOutputInputMatrix = Utils.add(this.updateOutputInputMatrix,
                    Utils.dot(Utils.transposition(data.getDataMatrix()), outputIHMatrixDelta));

            this.updateOutptuHiddenMatrix = Utils.add(this.updateOutptuHiddenMatrix,
                    Utils.dot(Utils.transposition(prevoutputHiddenValue), outputIHMatrixDelta));

            // output x hidden matrix
            double[][] tempDelta = Utils.add(Utils.multiple(Utils.multiple(Utils.dot(outputMatrixDelta,
                    Utils.transposition(outputMatrix)), sigmoidActivation.activation(
                    Utils.add(Utils.dot(data.getDataMatrix(), outputInputMatrix),
                            Utils.dot(prevoutputHiddenValue, outputHiddenMatrix)))),
                    tanhActivation.activationDerivative(ceremoryHiddenValue)), futureCeremoryHiddenDelta);

            double[][] ceremoryIHMatrixDelta = Utils.multiple(Utils.multiple(tempDelta,
                    sigmoidActivation.activation(Utils.add(Utils.dot(data.getDataMatrix(),
                            inputInputMatrix), Utils.dot(prevoutputHiddenValue,
                                    inputHiddenMatrix)))), tanhActivation.activationDerivative(
                            Utils.add(Utils.dot(data.getDataMatrix(), ceremoryInputMatrix),
                                    Utils.dot(prevoutputHiddenValue, ceremoryHiddenMatrix))));

            futureCeremoryHiddenDeltas.add(ceremoryIHMatrixDelta);

            this.updateCeremoryInputMatrix = Utils.add(this.updateCeremoryInputMatrix,
                    Utils.dot(Utils.transposition(data.getDataMatrix()), ceremoryIHMatrixDelta));

            this.updateCeremoryHiddenMatrix = Utils.add(this.updateCeremoryHiddenMatrix,
                    Utils.dot(Utils.transposition(prevCeremoryHiddenValue), ceremoryIHMatrixDelta));

            // output x hidden matrix
            double[][] inputIHMatrixDelta = Utils.multiple(Utils.multiple(tempDelta,
                    tanhActivation.activation(Utils.add(Utils.dot(
                            data.getDataMatrix(), ceremoryInputMatrix),
                            Utils.dot(prevoutputHiddenValue, ceremoryHiddenMatrix)))),
                    sigmoidActivation.activationDerivative(Utils.add(Utils.dot(
                            data.getDataMatrix(), inputInputMatrix), Utils.dot(
                                    prevoutputHiddenValue, inputHiddenMatrix))));

            this.updateInputInputMatrix = Utils.add(this.updateInputInputMatrix,
                    Utils.dot(Utils.transposition(data.getDataMatrix()), inputIHMatrixDelta));

            this.updateInputHiddenMatrix = Utils.add(this.updateInputHiddenMatrix,
                    Utils.dot(Utils.transposition(prevoutputHiddenValue), inputIHMatrixDelta));

            // output x hidden matrix
            double[][] forgetIHMatrixDelta = Utils.multiple(Utils.multiple(tempDelta,
                    prevCeremoryHiddenValue), sigmoidActivation.activationDerivative(
                            Utils.add(Utils.dot(data.getDataMatrix(), forgetInputMatrix),
                                    Utils.dot(prevoutputHiddenValue, forgetHiddenMatrix))));

            this.updateForgetInputMatrix = Utils.add(this.updateForgetInputMatrix,
                    Utils.dot(Utils.transposition(data.getDataMatrix()), forgetIHMatrixDelta));

            this.updateForgetHiddenMatrix = Utils.add(this.updateForgetHiddenMatrix,
                    Utils.dot(Utils.transposition(prevoutputHiddenValue), forgetIHMatrixDelta));

            futureCount++;

        }

        outputHiddenValues.clear();
        ceremoryHiddenValues.clear();
        outputs.clear();

    }

    private void updateParameters() {
        this.forgetInputMatrix = Utils.add(this.forgetInputMatrix,
                Utils.dot(updateForgetInputMatrix, -1 * learningRate));

        this.inputInputMatrix = Utils.add(this.inputInputMatrix,
                Utils.dot(updateInputInputMatrix, -1 * learningRate));

        this.ceremoryInputMatrix = Utils.add(this.ceremoryInputMatrix,
                Utils.dot(updateCeremoryInputMatrix, -1 * learningRate));

        this.outputInputMatrix = Utils.add(this.outputInputMatrix,
                Utils.dot(updateOutputInputMatrix, -1 * learningRate));

        this.forgetHiddenMatrix = Utils.add(this.forgetHiddenMatrix,
                Utils.dot(updateForgetHiddenMatrix, -1 * learningRate));

        this.inputHiddenMatrix = Utils.add(this.inputHiddenMatrix,
                Utils.dot(updateInputHiddenMatrix, -1 * learningRate));

        this.ceremoryHiddenMatrix = Utils.add(this.ceremoryHiddenMatrix,
                Utils.dot(updateCeremoryHiddenMatrix, -1 * learningRate));

        this.outputHiddenMatrix = Utils.add(this.outputHiddenMatrix,
                Utils.dot(updateOutptuHiddenMatrix, -1 * learningRate));

        this.outputMatrix = Utils.add(this.outputMatrix,
                Utils.dot(updateOutputMatrix, -1 * learningRate));

        Utils.matrixClear(updateForgetInputMatrix);
        Utils.matrixClear(updateInputInputMatrix);
        Utils.matrixClear(updateCeremoryInputMatrix);
        Utils.matrixClear(updateOutputInputMatrix);
        Utils.matrixClear(updateForgetHiddenMatrix);
        Utils.matrixClear(updateInputHiddenMatrix);
        Utils.matrixClear(updateCeremoryHiddenMatrix);
        Utils.matrixClear(updateOutptuHiddenMatrix);
        Utils.matrixClear(updateOutputMatrix);
    }

    public void train(int trainNum, int batchSize, List<Data> dataList) {
        for (int i = 0; i < trainNum; i++) {
            forwardCompute(dataList, true);
            backwardCompute(dataList);
            if ((i % (batchSize + 1)) == 0) {
                updateParameters();
            }
        }
    }

    public void predict(List<Data> dataList) {
        forwardCompute(dataList, true);
    }

}

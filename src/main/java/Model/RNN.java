/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model;
import Utils.*;
import java.util.ArrayList;
import java.util.List;
/**
 * 这是一个普通的循环神经网络
 * @author gzzengzihang
 */
public class RNN {
    private int nodeNum;
    private int inputNum;
    private int hiddenNum;
    private int outputNum;
    private double learningRate;
    private boolean ifSequence;
    private Activation activation;
    
    private double[][] inputMatrix;
    private double[][] hiddenMatrix;
    private double[][] outputMatrix;
    
    // hidden value
    private List<double[][]> beforeHiddenLayerValues;
    private List<double[][]> hiddenLayerValues;
    private List<double[][]> outputLayerValues;
    private List<double[][]> futureHiddenLayerDeltas;
    private List<double[][]> errors;
    private double[][] updateInputMatrix;
    private double[][] updateHiddenMatrix;
    private double[][] updateOutputMatrix;
    
    // loss function ==> 1/2 * (y - d)^2
    
    public RNN(int nodeNum, int inputNum, int hiddenNum, int outputNum, 
            double learningRate, boolean ifSequence){
        this.nodeNum = nodeNum;
        this.inputNum = inputNum;
        this.hiddenNum = hiddenNum;
        this.outputNum = outputNum;
        this.ifSequence = ifSequence;
        this.learningRate = learningRate;
        
        this.activation = new Activation("sigmoid");
        
        this.inputMatrix = Utils.randomMatrix(inputNum, hiddenNum);
        this.hiddenMatrix = Utils.randomMatrix(hiddenNum, hiddenNum);
        this.outputMatrix = Utils.randomMatrix(hiddenNum, outputNum);
        
        this.updateInputMatrix = new double[inputNum][hiddenNum];
        this.updateHiddenMatrix = new double[hiddenNum][hiddenNum];
        this.updateOutputMatrix = new double[hiddenNum][outputNum];
        
        this.beforeHiddenLayerValues = new ArrayList<>();
        this.hiddenLayerValues = new ArrayList<>();
        this.outputLayerValues = new ArrayList<>();
        this.futureHiddenLayerDeltas = new ArrayList<>();
        this.errors = new ArrayList<>();
        
        
    }
    
    private void forwardCompute(List<Data> dataList, boolean ifOutput){
        hiddenLayerValues.add(new double[1][hiddenNum]);
        List<double[][]> tempOutputs = new ArrayList<>();
        for (int i = 0; i < nodeNum; i++){
            Data data = dataList.get(i);
            double[][] prevHiddenLayerValue = hiddenLayerValues.get(i);
            double[][] beforeHiddenValue = Utils.add(Utils.dot(data.getDataMatrix(), 
                    inputMatrix), Utils.dot(prevHiddenLayerValue, hiddenMatrix));
            beforeHiddenLayerValues.add(beforeHiddenValue);
            double[][] hiddenLayerValue = activation.activation(beforeHiddenValue);
            hiddenLayerValues.add(hiddenLayerValue);
            double[][] output = new double[1][1];
            if (!ifSequence){
                if (i == nodeNum - 1){   
                    double[][] outputLayerValue = Utils.dot(hiddenLayerValue, outputMatrix);
                    outputLayerValues.add(outputLayerValue);
                    output = activation.activation(outputLayerValue);
                    errors.add(Utils.sub(data.getTarget(), output));
//                    outputs.add(output);
                
//                    System.out.println(output[0][0]);
                }
            }else{
                double[][] outputLayerValue = Utils.dot(hiddenLayerValue, outputMatrix);
                outputLayerValues.add(outputLayerValue);
                output = activation.activation(outputLayerValue);
                errors.add(Utils.sub(data.getTarget(), output));
//                outputs.add(output);
//                System.out.println(output[0][0]);
            }
            if (ifOutput){
                System.out.println(activation.activationLabel(output));
                tempOutputs.add(output);
            }
        }
    }
    
    private void backwardCompute(List<Data> dataList){
        int futureCount = 0;
        futureHiddenLayerDeltas.add(new double[1][hiddenNum]);
        
        for (int i = nodeNum - 1; i >= 0; i--){
            Data data = dataList.get(i);
            double[][] beforeHiddenLayerValue = beforeHiddenLayerValues.get(i);
            double[][] hiddenLayerValue = hiddenLayerValues.get(i + 1);
            double[][] prevHiddenLayerValue = hiddenLayerValues.get(i);
            double[][] futureHiddenLayerDelta = futureHiddenLayerDeltas.get(futureCount);
            double[][] outputMatrixDelta;
            if (!ifSequence){
                if (i == nodeNum - 1){
                    double[][] error = errors.get(0);
//                    double[][] error = Utils.sub(data.getTarget(), outputs.get(0));
                    double[][] outputLayerValue = outputLayerValues.get(0);
                    // output x output matrix
                    outputMatrixDelta = Utils.dot(Utils.dot(error, 
                            activation.activationDerivative(outputLayerValue)), -1);
                    // hidden x output matrix
                    updateOutputMatrix = Utils.add(updateOutputMatrix, 
                            Utils.transposition(Utils.dot(outputMatrixDelta, hiddenLayerValue)));
                }else{
                    outputMatrixDelta = new double[1][1];
                }
            }else{
                double[][] error = errors.get(i);
//                double[][] error = Utils.sub(data.getTarget(), outputs.get(i));
                double[][] outputLayerValue = outputLayerValues.get(i);
                // output x output matrix
                outputMatrixDelta = Utils.dot(Utils.dot(error,
                        activation.activationDerivative(outputLayerValue)), -1);
                // hidden x output matrix
                updateOutputMatrix = Utils.add(updateOutputMatrix,
                        Utils.transposition(Utils.dot(outputMatrixDelta, hiddenLayerValue)));
                
            }
            // ouput x hidden matrix
            double[][] hiddenMatrixDelta = Utils.multiple(Utils.add(Utils.dot(outputMatrixDelta,
                    Utils.transposition(outputMatrix)), futureHiddenLayerDelta),
                    activation.activationDerivative(beforeHiddenLayerValue));
            // hidden x hidden matrix
            updateHiddenMatrix = Utils.add(updateHiddenMatrix, Utils.dot(
                    Utils.transposition(hiddenMatrixDelta), prevHiddenLayerValue));

            updateInputMatrix = Utils.add(updateInputMatrix, Utils.dot(
                    Utils.transposition(data.getDataMatrix()), hiddenMatrixDelta));

            futureHiddenLayerDeltas.add(hiddenMatrixDelta);
            futureCount++;
        }
        beforeHiddenLayerValues.clear();
        hiddenLayerValues.clear();
        outputLayerValues.clear();
        futureHiddenLayerDeltas.clear();
        errors.clear();
    }
    
    private void updateParameters(){
        outputMatrix = Utils.add(outputMatrix, Utils.dot(updateOutputMatrix, learningRate));
        hiddenMatrix = Utils.add(hiddenMatrix, Utils.dot(updateHiddenMatrix, learningRate));
        inputMatrix = Utils.add(inputMatrix, Utils.dot(updateInputMatrix, learningRate));
        Utils.matrixClear(updateOutputMatrix);
        Utils.matrixClear(updateHiddenMatrix);
        Utils.matrixClear(updateInputMatrix);

    }
    
    public void train(int trainNum, int batchSize, List<Data> dataList){
        for (int i = 0; i < trainNum; i++){
            forwardCompute(dataList, false);
            backwardCompute(dataList);
            if (((i + 1) % batchSize) == 0)
                updateParameters();
        }
    }
    
    public void predict(List<Data> dataList){
        forwardCompute(dataList, true);
    }
    

    
    public static void main(String[] args){
        String dataPath = "src/main/java/Resources/data.txt";
        List<Data> dataList = FileIO.readData(dataPath, 1, 3);
        RNN rnn = new RNN(4, 3, 16, 1, -0.001, true);
        rnn.train(100000, 128, dataList);
        rnn.predict(dataList);
    }

}

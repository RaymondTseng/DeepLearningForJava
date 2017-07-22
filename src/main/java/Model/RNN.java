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
    private static final String modelName = "RNN";
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
    private List<double[][]> outputHiddenLayerValues;
    private List<double[][]> outputs;
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
        
        this.outputHiddenLayerValues = new ArrayList<>();
        this.outputs = new ArrayList<>();
        
        
    }
    
    private void forwardCompute(List<Data> dataList, boolean ifOutput){
        outputHiddenLayerValues.add(new double[1][hiddenNum]);
        for (int i = 0; i < nodeNum; i++){
            Data data = dataList.get(i);
            double[][] prevHiddenLayerValue = outputHiddenLayerValues.get(i);
            double[][] outputHiddenLayerValue = activation.activation(Utils.add(
                    Utils.dot(data.getDataMatrix(), inputMatrix), Utils.dot(
                            prevHiddenLayerValue, hiddenMatrix)));
            outputHiddenLayerValues.add(outputHiddenLayerValue);
            if (!ifSequence){
                if (i == nodeNum - 1){
                    double[][] output = activation.activation(Utils.dot(outputHiddenLayerValue, outputMatrix));
                    outputs.add(output);
                }
            }else{
                double[][] output = activation.activation(Utils.dot(outputHiddenLayerValue, outputMatrix));
                outputs.add(output);
            }
        }
        
        if (ifOutput){
            for (double[][] output : outputs){
                System.out.println(output[0][0]);
            }
        }

    }
    
    private void backwardCompute(List<Data> dataList){
        int futureCount = 0;
        List<double[][]> futureHiddenLayerDeltas = new ArrayList<>();
        futureHiddenLayerDeltas.add(new double[1][hiddenNum]);
        for (int i = nodeNum - 1; i >= 0; i--){
            Data data = dataList.get(i);
            double[][] outputHiddenLayerValue = outputHiddenLayerValues.get(i + 1);
            double[][] prevOutputHiddenLayerValie = outputHiddenLayerValues.get(i);
            double[][] futureHiddenLayerDelta = futureHiddenLayerDeltas.get(futureCount);
            double[][] outputMatrixDelta;
            
            if (!ifSequence){
                if (i == nodeNum - 1){
                    double[][] error = Utils.sub(data.getTarget(), outputs.get(0));
                    
                    // output x output matrix
                    outputMatrixDelta = Utils.dot(Utils.dot(error, -1),
                            activation.activationDerivative(Utils.dot(outputHiddenLayerValue, outputMatrix)));

                    updateOutputMatrix = Utils.add(updateOutputMatrix, Utils.dot(
                            Utils.transposition(outputHiddenLayerValue), outputMatrixDelta));
                }else{
                    outputMatrixDelta = new double[1][1];
                }
            }else{
                double[][] error = Utils.sub(data.getTarget(), outputs.get(i));

                // output x output matrix
                outputMatrixDelta = Utils.dot(Utils.dot(error, -1),
                        activation.activationDerivative(Utils.dot(outputHiddenLayerValue, outputMatrix)));

                updateOutputMatrix = Utils.add(updateOutputMatrix, Utils.dot(
                        Utils.transposition(outputHiddenLayerValue), outputMatrixDelta));
            }
            
            // output x hidden matrix
            double[][] HIMatrixDelta = Utils.multiple(Utils.add(futureHiddenLayerDelta, 
                    Utils.dot(outputMatrixDelta, Utils.transposition(outputMatrix))), 
                    activation.activation(Utils.add(Utils.dot(data.getDataMatrix(), 
                            inputMatrix), Utils.dot(prevOutputHiddenLayerValie, hiddenMatrix))));
            
            futureHiddenLayerDeltas.add(HIMatrixDelta);
            
            updateHiddenMatrix = Utils.add(updateHiddenMatrix, Utils.dot(Utils.transposition(
                    prevOutputHiddenLayerValie), HIMatrixDelta));
            
            updateInputMatrix = Utils.add(updateInputMatrix, Utils.dot(Utils.transposition(
                    data.getDataMatrix()), HIMatrixDelta));
            
            futureCount ++;
            
        }
        outputHiddenLayerValues.clear();
        outputs.clear();
    }
    
    private void updateParameters(){
        outputMatrix = Utils.add(outputMatrix, Utils.dot(updateOutputMatrix, -1 * learningRate));
        hiddenMatrix = Utils.add(hiddenMatrix, Utils.dot(updateHiddenMatrix, -1 * learningRate));
        inputMatrix = Utils.add(inputMatrix, Utils.dot(updateInputMatrix, -1 * learningRate));
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

}

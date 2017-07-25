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
 * 这是一个普通的RNN + Attention
 * 这是一个生成模型
 * @author gzzengzihang
 */
public class RNNAttention {
    private static final String modelName = "RNN";
    private int nodeNum;
    private int inputNum;
    private int hiddenNum;
    private int outputNum;
    private double learningRate;
    private boolean ifSequence;
    private Activation tanhActivation;

    
    private double[][] inputMatrix;
    private double[][] hiddenMatrix;
    private double[][] outputMatrix;
    private double[][] attentionHiddenMatrix;
    private double[][] attetionTargetMatrix;
    private double attetionParameter;
    
    // hidden value
    private List<double[][]> outputHiddenLayerValues;
    private List<double[][]> outputs;
    private double[][] updateInputMatrix;
    private double[][] updateHiddenMatrix;
    private double[][] updateOutputMatrix;
    private double[][] updateAttentionHiddenMatrix;
    private double[][] updateAttentionTargetMatrix;
    private double updateAttentionParameter;
     
    // loss function ==> 1/2 * (y - d)^2
    
    public RNNAttention(int nodeNum, int inputNum, int hiddenNum, int outputNum, 
            double learningRate, boolean ifSequence){
        this.nodeNum = nodeNum;
        this.inputNum = inputNum;
        this.hiddenNum = hiddenNum;
        this.outputNum = outputNum;
        this.ifSequence = ifSequence;
        this.learningRate = learningRate;
        
        this.tanhActivation = new Activation("tanh");
        
        this.inputMatrix = Utils.randomMatrix(inputNum, hiddenNum);
        this.hiddenMatrix = Utils.randomMatrix(hiddenNum, hiddenNum);
        this.outputMatrix = Utils.randomMatrix(hiddenNum, outputNum);
        this.attentionHiddenMatrix = Utils.randomMatrix(1, hiddenNum);
        this.attetionTargetMatrix = Utils.randomMatrix(1, hiddenNum);
        this.attetionParameter = (Math.random() * 2) - 1;
        
        this.updateInputMatrix = new double[inputNum][hiddenNum];
        this.updateHiddenMatrix = new double[hiddenNum][hiddenNum];
        this.updateOutputMatrix = new double[hiddenNum][outputNum];
        this.updateAttentionHiddenMatrix = new double[1][hiddenNum];
        this.updateAttentionTargetMatrix = new double[1][hiddenNum];
        this.updateAttentionParameter = 0;
        
        this.outputHiddenLayerValues = new ArrayList<>();
        this.outputs = new ArrayList<>();
        
        
    }
    
    private void forwardCompute(Data data, boolean ifOutput){
        outputHiddenLayerValues.add(new double[1][hiddenNum]);
        for (int i = 0; i < nodeNum; i++){
            double[][] prevHiddenLayerValue = outputHiddenLayerValues.get(i);
            double[][] outputHiddenLayerValue = tanhActivation.activation(Utils.add(
                    Utils.dot(data.getDataMatrix(), inputMatrix), Utils.dot(
                            prevHiddenLayerValue, hiddenMatrix)));
            outputHiddenLayerValues.add(outputHiddenLayerValue);
        }
        
        double[][] targetHiddenValue = outputHiddenLayerValues.get(outputHiddenLayerValues.size() - 1);
        
        double[][] weightValues = new double[1][outputHiddenLayerValues.size() - 1];
        
        for (int i = 0; i < outputHiddenLayerValues.size() - 1; i++){
            double[][] weightValue = Utils.dot(tanhActivation.activation(Utils.add(Utils.dot(Utils.transposition(
                    attentionHiddenMatrix), outputHiddenLayerValues.get(i)), Utils.dot(
                            Utils.transposition(attetionTargetMatrix), targetHiddenValue))), attetionParameter);
            weightValues[0][i] = weightValue[0][0];
        }
        
        double[] result = ActivationFunction.softmax(weightValues[0]);
        if(ifOutput){
            int label = ActivationFunction.softmaxLabel(result);
            System.out.println(label);
        }
        
    }
    
    public void train(int trainNum, int batchSize, List<Data> dataList){
        for (Data data : dataList){
            
        }
    }
    
}

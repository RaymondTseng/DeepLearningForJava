/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model;
import Utils.Utils;
import java.util.ArrayList;
import java.util.List;
/**
 * 这是一个普通的LSTM
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
            double learningRate, boolean ifSequence){
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
    
    public void forwardCompute(List<Data> dataList, boolean ifOutput){
        this.outputHiddenValues.add(new double[1][hiddenNum]);
        this.ceremoryHiddenValues.add(new double[1][hiddenNum]);
        for (int i = 0; i < nodeNum; i++){
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
            double[][] outputValue = sigmoidActivation.activation(Utils.add(
                    Utils.dot(data.getDataMatrix(), outputInputMatrix), 
                    Utils.dot(prevOutputHiddenValue, outputHiddenMatrix)));
            outputValue = Utils.multiple(outputValue, tanhActivation.activation(ceremoryValue));
            double[][] output = sigmoidActivation.activation(Utils.dot(outputValue, outputMatrix));
            this.outputHiddenValues.add(outputValue);
            this.ceremoryHiddenValues.add(ceremoryValue);
            this.outputs.add(output);
            
        }
    }
    
    private void backwardCompute(List<Data> dataList){
        int futureCount = 0;
        for (int i = nodeNum - 1; i >= 0; i--){
            
        }
    }
}

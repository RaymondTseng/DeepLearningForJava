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
 *
 * @author gzzengzihang
 */
public class RNNNode {
    private double[][] inputMatrix;
    private double[][] hiddenMatrix;
    private double[][] outputMatrix;
    
    private double[][] inputUpdate;
    private double[][] hiddenUpdate;
    private double[][] outputUpdate;
    
    private double[][] hiddenNetValues;
    private double[][] outputValues;
    
    private Activation activationFunction;
    
    private double learningRate;
    
    public RNNNode(int inputNum, int hiddenNum, int outputNum, double learningRate){
        inputMatrix = Utils.randomMatrix(inputNum, hiddenNum);
        hiddenMatrix = Utils.randomMatrix(hiddenNum, hiddenNum);
        outputMatrix = Utils.randomMatrix(hiddenNum, outputNum);
        
        inputUpdate = new double[inputNum][hiddenNum];
        hiddenUpdate = new double[hiddenNum][hiddenNum];
        outputUpdate = new double[hiddenNum][outputNum];
        
        //默认激活函数为sigmoid
        activationFunction = new Activation("sigmoid");
        
        this.learningRate = learningRate;
    }
    public RNNNode(int inputNum, int hiddenNum, int outputNum, double learningRate, String activationFunctionName){
        this(inputNum, hiddenNum, outputNum, learningRate);
        activationFunction = new Activation(activationFunctionName);
    }
    
    public double[][] forwardCompute(double[][] x, double[][] y, double[][] lastOutput, boolean ifOutput){
        hiddenNetValues = activationFunction.activation(Utils.add(
                Utils.dot(x, this.inputMatrix), Utils.dot(lastOutput, this.hiddenMatrix)));
        Utils.dot(hiddenNetValues, this.outputMatrix);
        double[][] outputNetValues = activationFunction.activation(Utils.dot(hiddenNetValues, this.outputMatrix));
        
        outputValues = Utils.sub(y, outputNetValues);
        if (ifOutput){
            double value = outputValues[0][0];
            System.out.println(value);
            if (value < 0.5)
                System.out.println(0);
            else
                System.out.println(1);
        }
        return hiddenNetValues;
    }
    
    public double[][] backwardCompute(double[][] x, double[][] y, double[][] prevHiddenNetValues, double[][] futureHiddenNetDeltas){
        double[][] outputNetDeltas = Utils.multiple(Utils.sub(y, outputValues), 
                activationFunction.activationDerivative(outputValues));
        double[][] hiddenNetDeltas = Utils.multiple(Utils.add(Utils.dot(futureHiddenNetDeltas, 
                Utils.transposition(hiddenMatrix)), Utils.dot(outputNetDeltas, 
                        Utils.transposition(outputMatrix))), activationFunction.
                                activationDerivative(hiddenNetValues));
        
        outputUpdate = Utils.add(outputUpdate, Utils.dot(Utils.transposition
        (hiddenNetValues), outputNetDeltas));
        hiddenUpdate = Utils.add(hiddenUpdate, Utils.dot(Utils.transposition
        (prevHiddenNetValues), hiddenNetDeltas));
        inputUpdate = Utils.add(inputUpdate, Utils.dot(Utils.transposition(x), hiddenNetDeltas));
        
        outputMatrix = Utils.add(outputMatrix, Utils.dot(outputUpdate, learningRate));
        hiddenMatrix = Utils.add(hiddenMatrix, Utils.dot(hiddenUpdate, learningRate));
        inputMatrix = Utils.add(inputMatrix, Utils.dot(inputUpdate, learningRate));
        
        return hiddenNetDeltas;
    }   
    
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model;

import Utils.Utils;
import Utils.ActivationFunction;

/**
 *
 * @author gzzengzihang
 */
public class Activation {
    private String name;
    
    public Activation(String name){
        this.name = name;
    }
    
    public double[][] activation(double[][] x){
        switch(this.name){
            case "sigmoid":
                return sigmoid(x);
            default:
                return sigmoid(x);
        }
    }
    
    public double[][] activationDerivative(double[][] x){
        switch(this.name){
            case "sigmoid":
                return sigmoidDerivative(x);
            default:
                return sigmoidDerivative(x);
        }
    }
    
    private double[][] sigmoid(double[][] x){
        for (int i = 0; i < x.length; i++){
            for (int j = 0; j < x[i].length; j++){
                x[i][j] = ActivationFunction.sigmoid(x[i][j]);
            }
        }
        return x;
    }
    
    private double[][] sigmoidDerivative(double[][] x){
        for (int i = 0; i < x.length; i++){
            for (int j = 0; j < x[i].length; j++){
                x[i][j] = ActivationFunction.sigmoidDerivative(x[i][j]);
            }
        }
        return x;
    }
}

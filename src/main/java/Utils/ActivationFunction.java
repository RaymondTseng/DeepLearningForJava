/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Utils;

/**
 *
 * @author gzzengzihang
 */
public class ActivationFunction {
    
    public static double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }
    
    public static double sigmoidDerivative(double x){
        return 1 - (1 / (1 + Math.exp(-x)));
    }
    
    public static int sigmoidLabel(double x){
        if (x >= 0.5)
            return 1;
        return 0;
    }
    
    public static double tanh(double x){
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
    }
    
    public static double tanhDerivative(double x){
        return 1 - Math.pow(tanh(x), 2);
    }
    
    public static int tanhLabel(double x){
        if (x >= 0)
            return 1;
        return -1;
    }
    
    public static double[] softmax(double[] x){
        double[] result = new double[x.length];
        double sum = 0;
        for (int i = 0; i < x.length; i++){
            double num = x[i];
            result[i] = Math.exp(num);
            sum += result[i];
        }
        
        for (int i = 0 ; i < result.length; i++){
            result[i] /= sum;
        }
        return result;
    }
    
    public static int softmaxLabel(double[] x){
        int flag = 0;
        double max = x[0];
        for (int i = 1; i < x.length; i++){
            if (x[i] > max){
                flag = i;
                max = x[i];
            }
        }
        return flag;
    }
    
    private ActivationFunction(){
        
    }
}

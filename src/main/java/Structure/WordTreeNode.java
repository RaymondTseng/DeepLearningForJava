/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Structure;
import Utils.*;
/**
 *
 * @author administrator
 */
public class WordTreeNode<T> extends HuffmanTreeNode<T>{
    
    private double[][] parameters;
    
    public WordTreeNode(T data, int weight, int parameterNum) {
        super(data, weight);
        if (data != null)
            parameters = Utils.randomMatrix(1, parameterNum);
    }

    public double[][] getParameters() {
        return parameters;
    }

    public void setParameters(double[][] parameters) {
        this.parameters = parameters;
    }
    

    public double classify(double[][] x){
        double[][] output = Utils.dot(x, Utils.transposition(this.parameters));
        double prob = ActivationFunction.sigmoid(output[0][0]);
        return prob;
    }
    
}

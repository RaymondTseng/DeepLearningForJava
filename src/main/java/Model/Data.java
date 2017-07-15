/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model;

/**
 *
 * @author gzzengzihang
 */
public class Data {
    private double[][] dataMatrix;
    private double[][] target;
    
    public Data(double[][] daraMatrix, double target){
        this.dataMatrix = daraMatrix;
        this.target = new double[1][1];
        this.target[0][0] = target;
    }

    public double[][] getDataMatrix() {
        return dataMatrix;
    }

    public double[][] getTarget() {
        return target;
    }


    
    
}

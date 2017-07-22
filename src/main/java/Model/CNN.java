/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model;

import java.util.List;
import Utils.Utils;
import java.util.ArrayList;
/**
 * 这是一个普通的用于文本的CNN
 * @author raymondtseng
 */
public class CNN {
    private List<Integer> filterSize;
    private List<double[][]> filters;
    
    public CNN(List<Integer> filterSize){
        initFilters();
    }
    
    private void initFilters(){
        for (int size : filterSize){
            double[][] filter = Utils.randomMatrix(size, size);
            filters.add(filter);
        }
    }
    
    private void fowardCompute(List<Data> dataList){
        for (Data data : dataList){
            List<double[][]> convolutionResults = new ArrayList<>();
            
            // convolution layer
            for (double[][] filter : filters){
                double[][] convolutionResult = convolution(data.getDataMatrix(), filter);
                convolutionResults.add(convolutionResult);
            }
            
            //
        }
    }
    
    private double[][] convolution(double[][] dataMatrix, double[][] filter){
        if (dataMatrix.length < filter.length || dataMatrix[0].length < filter[0].length){
            return null;
        }
        int row = dataMatrix.length - filter.length + 1;
        int column = dataMatrix[0].length - filter[0].length + 1;
        double[][] result = new double[row][column];
        for (int i = 0; i < row; i++){
            for (int j = 0; j < column; j++){
                double[][] tempMatrix = Utils.copyMatrix(dataMatrix, i, i + filter.length, 
                        j, j + filter[0].length);
                result[i][j] = Utils.convolution(tempMatrix, filter);
            }
        }
        return result;
    }
}

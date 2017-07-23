/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model;

import Utils.FileIO;
import java.util.List;
import Utils.Utils;
import java.util.ArrayList;

/**
 * 这是一个普通的用于文本的CNN
 *
 * @author raymondtseng
 */
public class CNN {

    private List<double[][]> filters;
    private List<Integer> filterSizes;
    private int rowNum;
    private int columnNum;
    private int outputNum;
    private double learningRate;
    private Activation activation;

    private double[][] outputMatrix;

    private double[][] updateOutputMatrix;
    private List<double[][]> updateFilters;

    // hidden value
    private double[][] output;
    private double[][] maxPoolingLayerValue;
    private List<int[]> maxPoolingIndices;
    private List<double[][]> convolutionResults;

    // loss function 1 / 2 (y - d)^2
    public CNN(int rowNum, int columnNum, int outputNum, double learningRate, List<Integer> filterSizes) {
        this.rowNum = rowNum;
        this.columnNum = columnNum;
        this.outputNum = outputNum;
        this.learningRate = learningRate;
        this.filterSizes = filterSizes;
        this.activation = new Activation("sigmoid");

        initValues();

        this.output = new double[1][outputNum];
        this.maxPoolingLayerValue = new double[1][filters.size()];
        this.convolutionResults = new ArrayList<>();
        this.maxPoolingIndices = new ArrayList<>();
    }

    private void initValues() {
        if (filterSizes == null) {
            filterSizes = new ArrayList<>();
        }

        if (filterSizes.isEmpty()) {
            for (int i = 1; i <= rowNum; i++) {
                filterSizes.add(i);
            }
        }
        filters = new ArrayList<>();
        updateFilters = new ArrayList<>();
        for (int filterSize : filterSizes) {
            double[][] filter = Utils.randomMatrix(filterSize, columnNum);
            filters.add(filter);
            updateFilters.add(new double[filterSize][columnNum]);
        }
        this.outputMatrix = Utils.randomMatrix(filterSizes.size(), outputNum);
        this.updateOutputMatrix = new double[filterSizes.size()][outputNum];
    }

    private void fowardCompute(Data data, boolean ifOutput) {

        // convolution layer
        for (double[][] filter : filters) {
            double[][] convolutionResult = convolution(data.getDataMatrix(), filter);
            convolutionResults.add(convolutionResult);
        }

        // max-pooling layer
        for (int i = 0; i < convolutionResults.size(); i++) {
            double[][] convolutionResult = convolutionResults.get(i);
            int[][][] maxValueIndices = Utils.maxPooling(convolutionResult, 1 / (double) convolutionResult[0].length);
            maxPoolingLayerValue[0][i] = convolutionResult[maxValueIndices[0][0][0]][maxValueIndices[0][0][1]];
            maxPoolingIndices.add(maxValueIndices[0][0]);
        }

        // output layer
        output = activation.activation(Utils.dot(maxPoolingLayerValue, outputMatrix));
        
        if (ifOutput){
            System.out.println(output[0][0]);
        }

    }

    private void backwardCompute(Data data) {
        double[][] error = Utils.sub(data.getTarget(), output);

        double[][] outputMatrixDelta = Utils.dot(Utils.dot(error, -1), Utils.dot(
                maxPoolingLayerValue, outputMatrix));

        updateOutputMatrix = Utils.add(updateOutputMatrix, Utils.dot(
                Utils.transposition(maxPoolingLayerValue), outputMatrixDelta));

        for (int i = 0; i < filters.size(); i++) {
            double[][] filter = filters.get(i);
            int[] indices = maxPoolingIndices.get(i);
            double[][] convolutionDelta = Utils.dot(Utils.copyMatrix(
                    data.getDataMatrix(), indices[1], indices[1] + filter.length, 0,
                    filter[0].length), outputMatrixDelta[0][0]);

            double[][] updateFilter = updateFilters.get(i);
            updateFilter = Utils.add(updateFilter, convolutionDelta);
            updateFilters.set(i, updateFilter);
        }

        Utils.matrixClear(output);
        Utils.matrixClear(maxPoolingLayerValue);
        convolutionResults.clear();
        maxPoolingIndices.clear();

    }

    private double[][] convolution(double[][] dataMatrix, double[][] filter) {
        if (dataMatrix.length < filter.length || dataMatrix[0].length < filter[0].length) {
            return null;
        }
        int row = dataMatrix.length - filter.length + 1;
        int column = dataMatrix[0].length - filter[0].length + 1;
        double[][] result = new double[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                double[][] tempMatrix = Utils.copyMatrix(dataMatrix, i, i + filter.length,
                        j, j + filter[0].length);
                result[i][j] = Utils.convolution(tempMatrix, filter);
            }
        }
        return Utils.transposition(result);
    }

    public void train(int trainNum, int batchSize, List<Data> dataList) {
        if (dataList == null) {
            return;
        }

        for (int i = 0; i < trainNum; i++) {
            for (Data data : dataList) {
                fowardCompute(data, true);
                backwardCompute(data);
                if ((i + 1) % batchSize == 0) {
                    updateParameters();
                }
            }

        }
    }

    private void updateParameters() {
        outputMatrix = Utils.add(outputMatrix, Utils.dot(updateOutputMatrix, -1 * learningRate));
        Utils.matrixClear(updateOutputMatrix);
        for (int i = 0; i < filters.size(); i++) {
            double[][] filter = filters.get(i);
            filter = Utils.add(filter, Utils.dot(updateFilters.get(i), -1 * learningRate));
            filters.set(i, filter);
            Utils.matrixClear(updateFilters.get(i));
        }
    }

    public static void main(String[] args) {
        String dataPath = "src/main/java/Resources/cnn_data.txt";
        List<Data> dataList = FileIO.readData(dataPath, 4, 3);
        CNN cnn = new CNN(4, 3, 1, 0.001, null);
        cnn.train(100000, 100, dataList);

    }
}

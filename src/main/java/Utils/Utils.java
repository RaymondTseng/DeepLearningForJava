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
public class Utils {
    public static double[][] randomMatrix(int row, int column){
        double[][] matrix = new double[row][column];
        for (int i = 0; i < row; i++){
            for (int j = 0; j < column; j++){
                matrix[i][j] = (Math.random() * 2) - 1;
            }
        }
        return matrix;
    }
    
    public static double[][] transposition(double[] array){
        double[][] result = new double[array.length][1];
        for (int i = 0; i < array.length; i++){
            result[i][1] = array[i];
        }
        return result;
    }
    
    public static double[][] transposition(double[][] matrix){
        double[][] result = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++){
            for (int j = 0; j < matrix[i].length; j++){
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    public static double[][] multiple(double[][] matrix1, double[][] matrix2){
        if (matrix1.length != matrix2.length || matrix1[0].length != matrix2[0].length)
            return null;
        double[][] result = new double[matrix1.length][matrix1[0].length];
        for (int i = 0; i < matrix1.length; i++){
            for (int j = 0; j < matrix1[i].length; j++){
                result[i][j] = matrix1[i][j] * matrix2[i][j];
            }
        }
        return result;
    }
    
    public static double[][] dot(double[][] matrix1, double[][] matrix2){
        if (matrix1[0].length != matrix2.length)
            return null;
        double[][] result = new double[matrix1.length][matrix2[0].length];
        for (int i = 0; i < matrix1.length; i++){
            double[] temp1 = new double[matrix1[i].length];
            for (int j = 0; j < matrix1[i].length; j++){
                temp1[j] = matrix1[i][j];
            }
            double[] temp2 = new double[matrix2.length];
            for (int j = 0; j < matrix2[0].length; j++){
                for (int z = 0; z < matrix2.length; z++){
                    temp2[z] = matrix2[z][j];
                }
                double sum = 0;
                for (int z = 0; z < temp1.length; z++){
                    sum += temp1[z] * temp2[z];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }
    
    public static double[][] dot(double[][] matrix, double num){
        double[][] result = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++){
            for (int j = 0; j < matrix[i].length; j++){
                result[i][j] = matrix[i][j] * num;
            }
        }
        return result;
    }
    
    public static double[][] add(double[][] matrix1, double[][] matrix2){
        if (matrix1.length != matrix2.length || matrix1[0].length != matrix2[0].length)
            return null;
        double[][] result = new double[matrix1.length][matrix2[0].length];
        for (int i = 0; i < matrix1.length; i++){
            for (int j = 0; j < matrix1[i].length; j++){
                result[i][j] = matrix1[i][j] + matrix2[i][j];
            }
        }
        return result;
    }
    
    public static double[][] sub(double[][] matrix1, double[][] matrix2){
        if (matrix1.length != matrix2.length || matrix1[0].length != matrix2[0].length)
            return null;
        double[][] result = new double[matrix1.length][matrix2[0].length];
        for (int i = 0; i < matrix1.length; i++){
            for (int j = 0; j < matrix1[i].length; j++){
                result[i][j] = matrix1[i][j] - matrix2[i][j];
            }
        }
        return result;
    }
    
    public static double[][] add(double[][] matrix, double num){
        double[][] result = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++){
            for (int j = 0; j < matrix[i].length; j++){
                result[i][j] = matrix[i][j] + num;
            }
        }
        return result;
    }
    
    public static double[][] add(double[][] matrix, double[] nums){
        if (nums.length != matrix.length)
            return null;
        double[][] result = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < nums.length; i++){
            double num = nums[i];
            for (int j = 0; j < matrix[i].length; j++){
                result[i][j] = matrix[i][j] + num;
            }
        }
        return matrix;
        
    }
    
    
    public static double getSumFromMatrix(double[][] matrix){
        double matrixSum = 0;
        for (int i = 0; i < matrix.length; i++){
            for (int j = 0; j < matrix[i].length; j++){
                matrixSum += matrix[i][j];
            }
        }
        return matrixSum;
    }
    
    public static void matrixClear(double[][] matrix){
        for (int i = 0; i < matrix.length; i++){
            for (int j = 0; j < matrix[i].length; j++){
                matrix[i][j] = 0;
            }
        }
    }
    
    public static double[][] copyMatrix(double[][] matrix, int startRow, int endRow, 
            int startColumn, int endColumn){
        if (matrix == null){
            return null;
        }
        if (startRow > endRow || startColumn > endColumn){
            return null;
        }
        if (startRow < 0 || endRow > matrix.length){
            return null;
        }
        if (startColumn < 0 || endColumn > matrix[0].length){
            return null;
        }
        double[][] result = new double[endRow - startRow][endColumn - startColumn];
        for (int i = startRow; i < endRow; i++){
            for (int j = startColumn; j < endColumn; j++){
                result[i - startRow][j - startColumn] = matrix[i][j];
            }
        }
        return result;
    }
    
    public static double convolution(double[][] matrix, double[][] filter){
        double result = 0;
        if (matrix.length != filter.length || matrix[0].length != filter[0].length)
            return result;
        for (int i = 0; i < matrix.length; i++){
            for (int j = 0; j < matrix[0].length; j++){
                result += matrix[i][j] * filter[i][j];
            }
        }
        return result;
    }
    
    public static int[][][] maxPooling(double[][] matrix, double rate){
        if (matrix == null)
            return null;
        int elementNum = matrix.length * matrix[0].length;
        int row = (int) (matrix.length * rate);
        row = row > 0 ? row : 1;
        int column = (int) (matrix[0].length * rate);
        column = column > 0 ? column : 1;
        int rowSize = (int)Math.round((double) matrix.length / row);
        int columnSize = (int)Math.round((double) matrix[0].length / column);
        int[][][] result = new int[row][column][2];
        for (int i = 0; i < row; i++){
            for (int j = 0; j < column; j++){
                int startRow = i * rowSize;
                int startColumn = j * columnSize;
                result[i][j] = findMaxValueInMatrix(matrix, startRow, 
                        startRow + rowSize, startColumn, startColumn + columnSize);
            }
        }
        return result;
    }
    
    public static int[] findMaxValueInMatrix(double[][] matrix, int startRow, 
            int endRow, int startColumn, int endColumn){
        int[] indices = new int[2];
        double max = matrix[startRow][startColumn];
        for (int i = startRow; i < endRow; i++){
            for (int j = startColumn; j < endColumn; j++){
                if (matrix[i][j] > max){
                    max = matrix[i][j];
                    indices[0] = i;
                    indices[1] = j;
                }
            }
        }
        return indices;
    }
    
    public static void main(String[] args){
        double[][] m1 = {{2,2,2},{3,3,3}};
        double[][] m2 = {{1,3,4},{2,2,5},{3,1,6}};
        dot(m1, m2);
    }

}

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
 * 这是一个普通的循环神经网络
 * @author gzzengzihang
 */
public class RNN {
    private List<RNNNode> nodeList;
    private int nodeNum;
    private int inputNum;
    private int hiddenNum;
    private int outputNum;
    private int trainNum;
    private boolean ifSequence;
    private List<double[][]> hiddenNetValues;
    
    public RNN(int nodeNum, int inputNum, int hiddenNum, int outputNum, double 
            learningRate, int trainNum, boolean ifSequence){
        this.nodeNum = nodeNum;
        this.inputNum = inputNum;
        this.hiddenNum = hiddenNum;
        this.outputNum = outputNum;
        this.trainNum = trainNum;
        this.ifSequence = ifSequence;
        
        nodeList = new ArrayList<>();
        hiddenNetValues = new ArrayList<>();
        for(int i = 0; i < nodeNum; i++){
            nodeList.add(new RNNNode(inputNum, hiddenNum, outputNum, learningRate));
        }
        
    }
    
    private void forwardCompute(List<Data> dataList){
        hiddenNetValues.add(new double[1][this.hiddenNum]);
        for (int i = 0; i < nodeNum; i++){
            Data data = dataList.get(i);
            RNNNode currentNode = nodeList.get(i);
            double[][] lastHiddenValues = hiddenNetValues.get(i);
            hiddenNetValues.add(currentNode.forwardCompute(data.getDataMatrix(), 
                        data.getTarget(), lastHiddenValues, false));
        }
    }
    
    private void backwardCompute(List<Data> dataList){
        List<double[][]> futureHiddenNetDeltas = new ArrayList<>();
        futureHiddenNetDeltas.add(new double[1][this.hiddenNum]);
        int futureCount = 0;
        for (int i = nodeNum - 1; i >= 0; i--){
            Data data = dataList.get(i);
            RNNNode currentNode = nodeList.get(i);
            double[][] prevHiddenValue = hiddenNetValues.get(i);
            double[][] futureHiddenNetDelta = futureHiddenNetDeltas.get(futureCount);
            futureHiddenNetDeltas.add(currentNode.backwardCompute(data.getDataMatrix(), data.getTarget(),
                    prevHiddenValue, futureHiddenNetDelta));
            futureCount ++;
        }
        futureHiddenNetDeltas.clear();
        hiddenNetValues.clear();
    }
    
    
    
    public void train(List<Data> dataList){
        for (int i = 0; i < trainNum; i++){
            forwardCompute(dataList);
            backwardCompute(dataList);
        }
    }
    
    public void predict(List<Data> dataList){
        hiddenNetValues.add(new double[1][this.hiddenNum]);
        for (int i = 0; i < nodeNum; i++){
            Data data = dataList.get(i);
            RNNNode currentNode = nodeList.get(i);
            double[][] lastHiddenValues = hiddenNetValues.get(i);
            if(ifSequence)
                hiddenNetValues.add(currentNode.forwardCompute(data.getDataMatrix(), 
                            data.getTarget(), lastHiddenValues, true));
            else{
                if (i == nodeNum - 1) {
                    hiddenNetValues.add(currentNode.forwardCompute(data.getDataMatrix(),
                            data.getTarget(), lastHiddenValues, true));
                } else {
                    hiddenNetValues.add(currentNode.forwardCompute(data.getDataMatrix(),
                            data.getTarget(), lastHiddenValues, false));
                }
            }
        }
    }
    
    public static void main(String[] args){
        String dataPath = "src/Resources/data.txt";
        List<Data> dataList = FileIO.readData(dataPath, 1, 3);
        RNN rnn = new RNN(dataList.size(), 3, 16, 1, 0.01, 10, true);
        rnn.train(dataList);
        rnn.predict(dataList);
    }
    
    
    
    
}

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Demo;

import Model.Data;
import Model.RNN;
import Utils.FileIO;
import java.util.List;

/**
 *
 * @author gzzengzihang
 */
public class RNNDemo {
    public static void main(String[] args){
        String dataPath = "src/main/java/Resources/data.txt";
        List<Data> dataList = FileIO.readData(dataPath, 1, 3);
        RNN rnn = new RNN(4, 3, 16, 1, 0.001, true);
        rnn.train(100000, 128, dataList);
        rnn.predict(dataList);
    }
}

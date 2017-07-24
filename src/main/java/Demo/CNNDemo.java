/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Demo;

import Model.CNN;
import Model.Data;
import Utils.FileIO;
import java.util.List;

/**
 *
 * @author gzzengzihang
 */
public class CNNDemo {
    public static void main(String[] args) {
        String dataPath = "src/main/java/Resources/cnn_data.txt";
        List<Data> dataList = FileIO.readData(dataPath, 4, 3);
        CNN cnn = new CNN(4, 3, 1, 0.001, null);
        cnn.train(10000, 1, dataList);
    }
}

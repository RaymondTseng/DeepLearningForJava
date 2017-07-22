/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model;

import Structure.*;
import java.util.*;
import Utils.Utils;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.Buffer;

/**
 *
 * @author administrator
 */
public class Word2Vec {
    // model basic information
    private static String modelName = "CBOW";
    private static String methodName = "HierarchicalSoftmax";
    
    //data
    private List<List<String>> sentences;
    
    //parameters
    private int size;
    private double alpha;
    private int window;
    private int minCount;
    private int trainNum;
    
    // hidden varibles
    private Map<String, Integer> wordDict;
    private Map<String, double[][]> wordEmbeddings;
    private List<String> words;
    private Map<String, Integer> word2Index;
    private HuffmanTree wordTree;
    
    public Word2Vec(List<List<String>> sentences, int size, double alpha, int window, int minCount){
        
        this.sentences = sentences;
        
        this.size = size;
        this.alpha = alpha;
        this.window = window;
        this.minCount = minCount;
        this.trainNum = 5;
        
        this.wordDict = new HashMap<>();
        this.wordEmbeddings = new HashMap<>();
        this.words = new ArrayList<>();
        this.word2Index = new HashMap<>();
        
        initVaribles();
        train();
        
    }
    
    private void initVaribles(){
        System.out.println("build word dict...");
        buildWordDict();
        System.out.println("build huffman tree...");
        this.wordTree = new HuffmanTree(wordDict);
        this.wordTree.createWordTree(wordDict, size);
        for (Map.Entry entry : wordDict.entrySet()){
            String word = (String) entry.getKey();
            int value = (int) entry.getValue();
            if (value > minCount)
                wordEmbeddings.put(word, Utils.randomMatrix(1, size));
        }
    }
    
    private void buildWordDict(){
        for (List<String> sentence : sentences){
            for (String word : sentence){
                if(wordDict.containsKey(word))
                    wordDict.put(word, wordDict.get(word) + 1);
                else{
                    words.add(word);
                    word2Index.put(word, words.size() - 1);
                    wordDict.put(word, 1);
                }
            }
        }
    }
    
    private void train(){
       System.out.println("training...");
       for (int t = 1; t <= trainNum; t++){
           double processCount = 0;
           for (List<String> sentence : sentences) {
               for (int i = 0; i < sentence.size(); i++) {
                   String word = sentence.get(i);
                   int leftIndex = i - window < 0 ? 0 : i - window;
                   int rightIndex = i + window >= sentence.size() ? sentence.size() - 1 : i + window;
                   double[][] embeddingSum = new double[1][size];
                   List<String> contextWords = new ArrayList<>();
                   for (int j = leftIndex; j <= rightIndex; j++) {
                       if (j != i) {
                           String contextWord = sentence.get(j);
                           double[][] contextWordEmbedding = wordEmbeddings.get(contextWord);
                           if (contextWordEmbedding != null) {
                               embeddingSum = Utils.add(embeddingSum, wordEmbeddings.get(contextWord));
                               contextWords.add(contextWord);
                           }
                           compute(embeddingSum, word, contextWords);
                       }
                   }
               }
               if (processCount % 1000 == 0) {
                   System.out.println("epoch:" + t + "," + 
                           "processing:" + String.format("%.2f", processCount / sentences.size()) 
                           + "% sentences");
               }
               processCount++;
           }
       }
    }
    
    private void compute(double[][] embeddingSum, String word, List<String> contextWords){
        Map<String, String> huffmanCode = wordTree.getHuffmanCode();
        WordTreeNode currentNode = (WordTreeNode)wordTree.getRoot();
        String code = huffmanCode.get(word);
        double[][] e = new double[1][size];
        for (int i = 0; i < code.length(); i++){
            int d = Integer.valueOf(String.valueOf(code.charAt(i)));
            double prob = currentNode.classify(embeddingSum);
            double g = alpha * (1 - d - prob);
            e = Utils.add(e, Utils.dot(currentNode.getParameters(), g));
            currentNode.setParameters(Utils.add(currentNode.getParameters(), Utils.dot(embeddingSum, g)));
            if (d == 1)
                currentNode = (WordTreeNode) currentNode.getLeftChild();
            else
                currentNode = (WordTreeNode) currentNode.getRightChild();
        }
        for (String contextWord : contextWords){
            double[][] v = Utils.add(wordEmbeddings.get(contextWord), e);
            wordEmbeddings.put(contextWord, v);
        }
    }
    
    public void save(String path){
        BufferedWriter bw = null;
        try{
            bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(path)));
            for (Map.Entry entry : this.wordEmbeddings.entrySet()){
                String word = (String) entry.getKey();
                double[] embedding = ((double[][]) entry.getValue())[0];
                String temp = word;
                for (double num : embedding){
                    temp += " " + Double.toString(num);
                }
                bw.write(temp + "\n");
            }
            bw.close();
        }catch (Exception e){
            System.err.println("save model error!!");
        }
    }
}

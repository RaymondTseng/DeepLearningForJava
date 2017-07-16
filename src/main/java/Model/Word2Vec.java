/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model;

import Structure.*;
import java.util.*;
import Utils.Utils;

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
        this.trainNum = 10000;
        
        this.wordDict = new HashMap<>();
        this.wordEmbeddings = new HashMap<>();
    }
    
    private void initVaribles(){
        buildWordDict();
        this.wordTree = new HuffmanTree(wordDict);
        for (Map.Entry entry : wordDict.entrySet()){
            String word = (String) entry.getKey();
            wordEmbeddings.put(word, Utils.randomMatrix(1, size));
        }
    }
    
    private void buildWordDict(){
        for (List<String> sentence : sentences){
            for (String word : sentence){
                words.add(word);
                word2Index.put(word, words.size() - 1);
                if(wordDict.containsKey(word))
                    wordDict.put(word, wordDict.get(word) + 1);
                else
                    wordDict.put(word, 1);
            }
        }
    }
    
    private void train(){
       for (List<String> sentence : sentences){
           for (int i = 0; i < sentence.size(); i++){
               String word = sentence.get(i);
               int leftIndex = i - window < 0 ? 0 : i - window;
               int rightIndex = i + window >= sentence.size() ? sentence.size() - 1 : i + window;
               double[][] embeddingSum = new double[1][size];
               List<String> contextWords = new ArrayList<>();
               for (int j = leftIndex; j <= rightIndex; j++){
                   if (j != i){
                       String contextWord = sentence.get(j);
                       contextWords.add(contextWord);
                       embeddingSum = Utils.add(embeddingSum, wordEmbeddings.get(contextWord));
                       compute(embeddingSum, word, contextWords);
                   }
               }
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
            if (d == 0)
                currentNode = (WordTreeNode) currentNode.getLeftChild();
            else
                currentNode = (WordTreeNode) currentNode.getRightChild();
        }
        for (String contextWord : contextWords){
            double[][] v = Utils.add(wordEmbeddings.get(contextWord), e);
            wordEmbeddings.put(contextWord, v);
        }
    }
    
    public static void main(String[] args){
        
    }
}
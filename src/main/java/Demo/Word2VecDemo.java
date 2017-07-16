/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Demo;
import Model.Word2Vec;
import Structure.HuffmanTree;
import Structure.HuffmanTreeNode;
import Utils.FileIO;
import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.JiebaSegmenter.SegMode;
import com.huaban.analysis.jieba.SegToken;
import java.util.*;
        
/**
 *
 * @author raymondtseng
 */
public class Word2VecDemo {
    
    private static String path = "/media/raymondtseng/软件/nlp/数据/fenghuangwang";
    
    private static List<List<String>> segmentUseJieba(List<String> sentences){
        List<List<String>> sentencesWords = new ArrayList<>();
        JiebaSegmenter segmenter = new JiebaSegmenter();
        for (String sentence : sentences) {
            List<SegToken> temp = segmenter.process(sentence, SegMode.INDEX);
            List<String> words = new ArrayList<>();
            for (SegToken token : temp){
                if (!token.word.trim().equals(""))
                    words.add(token.word);
            }
            sentencesWords.add(words);
        }
        return sentencesWords;
    }
    
    public static void main(String[] args){
        System.out.println("read sentences...");
        List<String> sentences = FileIO.readSentences(path);
        System.out.println("segment...");
        List<List<String>> sentencesWords = segmentUseJieba(sentences);
        System.out.println("start word2vec...");
        Word2Vec w2v = new Word2Vec(sentencesWords, 200, 0.025, 5, 5);
        
        System.out.println("done!");

        
    }
    
    
}

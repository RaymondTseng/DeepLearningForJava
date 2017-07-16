/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Structure;

import java.util.*;
import Structure.HuffmanTreeNode;

/**
 *
 * @author administrator
 */
public class HuffmanTree<T> {
    private HuffmanTreeNode<T> root;
    private Map<T, String> huffmanCode;
    
    public HuffmanTree(Map<T, Integer> wordDict){
        huffmanCode = new HashMap<>();
    }
    
    public void createWordTree(Map<T, Integer> wordDict, int size){
        List<HuffmanTreeNode<T>> nodeList = new ArrayList<HuffmanTreeNode<T>>();
        for (Map.Entry<T, Integer> entry : wordDict.entrySet()){
            nodeList.add(new WordTreeNode<T>(entry.getKey(), entry.getValue()));
        }
        while (nodeList.size() > 1){
            Collections.sort(nodeList);
            HuffmanTreeNode<T> leftChild = nodeList.get(nodeList.size() - 1);
            HuffmanTreeNode<T> rightChild = nodeList.get(nodeList.size() - 2);
            WordTreeNode<T> parent = new WordTreeNode<T>(null, 
                    leftChild.getWeight() + rightChild.getWeight(), size);
            parent.setLeftChild(leftChild);
            parent.setRightChild(rightChild);
            nodeList.remove(leftChild);
            nodeList.remove(rightChild);
            nodeList.add(parent);
        }
        root = nodeList.get(0);
        buildHuffmanCode();
    }
    
    private void buildHuffmanCode(){
        visit(root, "");
    }
    
    private void visit(HuffmanTreeNode node, String code){
        if (node != null){
            if (node.getData() != null){
                huffmanCode.put((T) node.getData(), code);
            }else{
                visit(node.getLeftChild(), code + "1");
                visit(node.getRightChild(), code + "0");
            }
        }
    }

    public Map<T, String> getHuffmanCode() {
        return huffmanCode;
    }

    public HuffmanTreeNode<T> getRoot() {
        return root;
    }
     
}

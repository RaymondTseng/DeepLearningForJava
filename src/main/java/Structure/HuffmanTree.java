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
        root = createTree(wordDict);
        huffmanCode = new HashMap<>();
        buildHuffmanCode();
    }
    
    private HuffmanTreeNode<T> createTree(Map<T, Integer> wordDict){
        List<HuffmanTreeNode<T>> nodeList = new ArrayList<HuffmanTreeNode<T>>();
        for (Map.Entry<T, Integer> entry : wordDict.entrySet()){
            nodeList.add(new HuffmanTreeNode<T>(entry.getKey(), entry.getValue()));
        }
        while (nodeList.size() > 1){
            Collections.sort(nodeList);
            HuffmanTreeNode<T> leftChild = nodeList.get(nodeList.size() - 1);
            HuffmanTreeNode<T> rightChild = nodeList.get(nodeList.size() - 2);
            HuffmanTreeNode<T> parent = new HuffmanTreeNode<T>(null, 
                    leftChild.getWeight() + rightChild.getWeight());
            parent.setLeftChild(leftChild);
            parent.setRightChild(rightChild);
            nodeList.remove(leftChild);
            nodeList.remove(rightChild);
            nodeList.add(parent);
        }
        return nodeList.get(0);
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

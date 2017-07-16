/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Structure;

/**
 *
 * @author administrator
 */
public class HuffmanTreeNode<T> implements Comparable{
    private T data;
    private int weight;
    private HuffmanTreeNode<T> leftChild;
    private HuffmanTreeNode<T> rightChild;

    public HuffmanTreeNode(T data, int weight) {
        this.data = data;
        this.weight = weight;
    }

    public T getData() {
        return data;
    }

    public int getWeight() {
        return weight;
    }

    public HuffmanTreeNode<T> getLeftChild() {
        return leftChild;
    }

    public HuffmanTreeNode<T> getRightChild() {
        return rightChild;
    }

    public void setLeftChild(HuffmanTreeNode<T> leftChild) {
        this.leftChild = leftChild;
    }

    public void setRightChild(HuffmanTreeNode<T> rightChild) {
        this.rightChild = rightChild;
    }
    
    @Override
    public int compareTo(Object o) {
        HuffmanTreeNode node = (HuffmanTreeNode) o;
        if (node.getWeight() > weight) {
            return 1;
        } else if (node.getWeight() < weight) {
            return -1;
        } else {
            return 0;
        }
    }
}

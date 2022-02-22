# Bert-classification

使用HuggingFace开发的Transformers库，使用BERT模型实现中文文本分类（二分类或多分类）

首先直接利用`transformer.models.bert.BertForSequenceClassification()`实现文本分类

然后手动实现BertModel + FC 实现上边函数。其中可以方便的更改参数和结构

然后实验了论文中将bert最后四层进行concat再maxpooling的方法，

最后实现了bert + CNN实现文本分类



## Environment

`pytorch == 1.7.0`

`transformers == 4.5.1`

## DATASETS

使用苏神的中文评论情感二分类数据集

原始Github链接：https://github.com/bojone/bert4keras/tree/master/examples/datasets



## BERT

模型使用的是哈工大[chinese-bert-wwm](https://github.com/ymcui/Chinese-BERT-wwm),可以完全兼容BERT

下载：

```
git clone https://huggingface.co/hfl/chinese-bert-wwm
```



## HOW TO USE

1. (optional) 下载huggingface/transformer 库到根目录
2. 下载哈工大的预训练模型到根目录
3. 运行src/train.py
4. 如果要更换模型就改train.py的65-70行



## 结果

除了第一个实验dropout_bert是0.1，其余是0.2. 剩下参数都一样。

训练3个epoch

| 模型                                  | train/val acc | val acc | test acc | 链接                                                         |
| :------------------------------------ | ------------- | ------- | -------- | ------------------------------------------------------------ |
| 会用内建BertForSequenceClassification | 0.982         | 0.950   | 0.950    | [链接](https://github.com/karlhl/Bert-classification/blob/main/src/bert_CNN.py) |
| 自己实现Bert+fc 一层全连接层          | 0.982         | 0.948   | 0.954    | [链接](https://github.com/karlhl/Bert-classification/blob/main/src/bert_lr.py) |
| 将Bert最后四层相concat然后maxpooling  | 0.977         | 0.946   | 0.951    | [链接](https://github.com/karlhl/Bert-classification/blob/main/src/bert_lr_last4layer.py) |
| BERT+CNN                              | 0.984         | 0.947   | 0.955    | [链接](https://github.com/karlhl/Bert-classification/blob/main/src/bert_CNN.py) |

1. 官方的`transformer.models.bert.BertForSequenceClassification()`就是直接使用BertModel 再接一层全连接层实现的。第二个项目是为了方便自己修改网络结构，进行手动实现。效果差不多，可以自己修改接几层线形结构，但是实验了一层就够了。
2. 根据参考2的论文，将BERT最后四层的CLS向量concat然后取max pooling可以让bert在分类问题上有更好的效果。在THUNews上测试可以提高0.4%相比bert。已经很大了相比其他方法而言。
3. 我一直觉得bert后面接CNN和RNN等都不好，毕竟transformer就是改善这两类模型的，再接一层也好不到哪去。如果我理解不到位可以告诉我。我还实验了bert使用前四层的输出进行concat，效果acc也能达到0.80+，层数越深效果感觉真的不明显。bert+cnn/rnn等这个模型在参考3 中两年前就有人做过实验，写出来过，他实验的[效果](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch#%E6%95%88%E6%9E%9C)也是不如单纯的BERT。调了调cnn的大小，其实都差不多。。





## 参考

1. https://github.com/sun830910/Transformers_binary_classification
2. [How to Fine-Tune BERT for Text Classification（2019）](https://www.aclweb.org/anthology/P18-1031.pdf)
3. https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch


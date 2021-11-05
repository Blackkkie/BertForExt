# BertForExt
The implement of "Fine-tune BERT for Extractive Summarization".



#### 数据处理

1. 首先定义$f(*)$为一个截断操作。
2. 其次根据$f(T_{src})$以及$T_{tgt}$来获取$T_{label}$，主要思想就是，每次从所有句子中选取一句能使得ROUGE-1+ROUGE-2分数最高的句子。
3. 将$T_{label}$中正例数为0的样本删除（可选）。



#### 实验结果

| Version |                     Version_Description                      | data_process |   R-1   |   R-2   |   R-L   |
| :-----: | :----------------------------------------------------------: | :----------: | :-----: | :-----: | :-----: |
|  mine   | Oracle(先对文章根据max_pos及多余句进行删除，用paper的计算方法，限制选取的句子数3) |      /       | 0.54745 | 0.32038 | 0.50691 |
|  paper  |                              /                               |      /       | 0.5259  | 0.3124  | 0.4887  |
|  mine   |        Bert+Transformer(n_layer=2)+Linear Classifier         |      /       | 0.41482 | 0.19221 | 0.35111 |
|  paper  |                              /                               |      /       | 0.4325  | 0.2024  | 0.3963  |

### 

![image](https://user-images.githubusercontent.com/53401764/140505930-3bb05d3f-c0da-4f1f-9834-270fe54dda8c.png)

(未调参，在一个epoch下直接就拿出来测试了)

# BertForExt
The implement of "Text Summarization with Pretrained Encoders", which is just for Extractive Summarization.



| Version |       Version_Description       |                         data_process                         |   R-1   |   R-2   |   R-L   |
| :-----: | :-----------------------------: | :----------------------------------------------------------: | :-----: | :-----: | :-----: |
|    0    |           Oracle(R-1)           |                              /                               | 0.39298 | 0.22540 | 0.37062 |
|   0.1   |           Oracle(R-2)           |                              /                               | 0.48211 | 0.28783 | 0.45342 |
|   0.2   |         Oracle(R-1+R-2)         |                              /                               | 0.44194 | 0.25844 | 0.41612 |
|  0.21   |         Oracle(R-1+R-2)         |                   清除除字母和数字外的符号                   | 0.43875 | 0.25307 | 0.37295 |
|   0.3   |      Oracle(Official Code)      |                              /                               | 0.56430 | 0.33866 | 0.52432 |
|  Paper  |                /                |                              /                               | 0.5259  | 0.3124  | 0.4887  |
|    1    |    Bert+Transformer-layer(2)    | 句子长度<=50<br />文章长度<=512<br />(处理CLS Mask的时候出错了) | 0.39864 | 0.17343 | 0.36012 |
|   1.1   |      Lead-3（仅取前三句）       |                              /                               | 0.40071 | 0.17503 | 0.36204 |
|    2    |    Bert+Transformer-layer(2)    |            句子长度<=50<br />文章长度<=512<br />             | 0.39207 | 0.16960 | 0.35427 |
|    3    |     Bert+Linear Classifier      |            句子长度<=50<br />文章长度<=512<br />             | 0.39563 | 0.17049 | 0.35721 |
|         |                                 |                                                              |         |         |         |
|   4.1   | Bert+Linear Classifier(epoch=1) | (用0.3版本生成的label)<br />句子长度不作限制<br />文章长度<=512 | 0.28004 | 0.08295 | 0.24990 |
|   4.2   | Bert+Linear Classifier(epoch=2) | (用0.3版本生成的label)<br />句子长度不作限制<br />文章长度<=512 | 0.40068 | 0.17501 | 0.36202 |
|   4.3   | Bert+Linear Classifier(epoch=2) | (用0.3版本生成的label)<br />句子长度不作限制<br />文章长度<=512 | 0.40060 | 0.17495 | 0.36193 |

![image](https://user-images.githubusercontent.com/53401764/139586918-674edfc5-166c-4edf-85a3-535a2cc3735d.png)

目前存在预测的所有值都尽可能低（怀疑因为标签里面0的含量远超1的含量，需要再看看paper里面的代码是否研究解决了这个问题，还是说这单纯是训练问题）。


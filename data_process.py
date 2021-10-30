
import json
from tqdm import tqdm
import config

from transformers import BertTokenizerFast

# 使用jieba进行分词
def Rouge_1(model, reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***one-gram*** 一元模型

    grams_model = model.split(' ')
    grams_reference = reference.split(' ')  # 默认精准模式

    precision = 0
    recall = 0

    for x in grams_reference:
        if x in grams_model:
            recall = recall + 1

    for x in grams_model:
        if x in grams_reference:
            precision = precision + 1

    #
    recall = recall / len(grams_reference)
    precision = precision / len(grams_model)

    sums = recall + precision

    f = 2*recall*precision/sums if sums != 0 else 0

    return f


def Rouge_2(model, reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***Bi-gram***  2元模型
    grams_reference = reference.split(' ')  # 默认精准模式
    grams_model = model.split(' ')
    gram_2_model = []
    gram_2_reference = []

    precision = 0
    recall = 0

    for x in range(len(grams_model) - 1):
        gram_2_model.append(grams_model[x] + grams_model[x + 1])
    for x in range(len(grams_reference) - 1):
        gram_2_reference.append(grams_reference[x] + grams_reference[x + 1])

    for x in gram_2_model:
        if x in gram_2_reference:
            recall = recall + 1

    for x in gram_2_reference:
        if x in gram_2_model:
            precision = precision + 1

    if len(grams_model) - 1 == 0 or (len(grams_reference) - 1) == 0:
        return 0

    precision = precision / (len(grams_model) - 1)
    recall = recall / (len(grams_reference) - 1)

    sums = recall + precision
    f = 2 * recall * precision / sums if sums != 0 else 0

    return f


def Rouge(model, reference):
    print("rouge_1=" + str(Rouge_1(model, reference)))
    print("rouge_2=" + str(Rouge_2(model, reference)))

# Rouge("我的世界是光明的","光明给我的世界以力量")


# 注意去除单个句号的情况
def turn2doc(sent_list):
    doc = []
    for sent in sent_list:
        sentence = []
        for token in sent:
            sentence.append(token)

        # 去除用于分句的句号
        if len(sentence) == 1 and sentence[0] == '.':
            continue

        doc.append(' '.join(sentence))

    return doc


def process_src(kind='train'):

    with open('./data/ssplit_%s_src.json' % kind, 'r', encoding='utf-8') as file:
        with open('./data/process_%s_src.txt' % kind, 'w', encoding='utf-8') as file1:

            bar = tqdm(file.readlines(), '读取原文文本：')
            for line in bar:
                doc = turn2doc(json.loads(line.strip()))

                file1.write(json.dumps(doc) + '\n')

def process_tgt(kind='train'):

    with open('./data/ssplit_%s_tgt.json' % kind, 'r', encoding='utf-8') as file:
        with open('./data/process_%s_tgt.txt' % kind, 'w', encoding='utf-8') as file1:

            bar = tqdm(file.readlines(), '读取摘要文本：')
            for line in bar:
                doc = turn2doc(json.loads(line.strip()))
                file1.write(json.dumps(doc) + '\n')


def greedy_generate_labels():

    # 读取原文和摘要
    with open('./data/process_train_src.txt', 'r', encoding='utf-8') as f_src:
        with open('./data/process_train_tgt.txt', 'r', encoding='utf-8') as f_tgt:
            with open('./data/labels.txt', 'w', encoding='utf-8') as f_pdt:

                bar = tqdm(list(zip(f_src.readlines(), f_tgt.readlines())), '生成oracle及标签')

                for j_src, j_tgt in bar:

                    src = json.loads(j_src.strip())                 # 分句的形式
                    tgt = ' '.join(json.loads(j_tgt.strip()))       # 文档形式

                    # 当前分数
                    labels = [0 for _ in range(len(src))]
                    scorer = 0
                    doc = []

                    for i, sent in enumerate(src):

                        f1 = Rouge_2(' '.join(doc + [sent]), tgt)

                        if f1 > scorer:
                            doc.append(sent)
                            scorer = f1
                            labels[i] = 1

                    if not doc:
                        # 如果是空的，那就默认全部都需要
                        labels = [1 for _ in range(len(src))]
                        f_pdt.write(json.dumps(labels) + '\n')
                    else:
                        f_pdt.write(json.dumps(labels) + '\n')

# 对数据提前进行tokenize和截断
def tokenize_and_truncature():

    tokenizer = BertTokenizerFast.from_pretrained('./bert-base-uncased')

    src, tgt = [], []

    with open('./data/process_train_src.txt', 'r', encoding='utf-8') as file_src:
        with open('./data/labels.txt', 'r', encoding='utf-8') as file_tgt:

            bar = tqdm(list(zip(file_src.readlines(), file_tgt.readlines())), '读取原文文本和标签：')
            for src_line, tgt_line in bar:

                sentences = json.loads(src_line.strip())
                cut_sentences_id = []

                # 对于其中的每一句都进行如下操作
                # 根据最大句子长度压缩每一个句子x
                # 根据最大长度压缩句子（512

                article_len = 0

                for sentence in sentences:

                    # 截断句子长度
                    tokens = sentence.split()
                    tail = min(config.max_sent_len, len(tokens))
                    sentence_id = tokenizer.encode(' '.join(['[CLS]'] + tokens[:tail] + ['[SEP]']), add_special_tokens=False)

                    # 防止总长度超过max_position_embedding
                    if article_len + len(sentence_id) > config.max_position_embeddings:
                        break

                    article_len += len(sentence_id)
                    cut_sentences_id.append(sentence_id)

                src.append(cut_sentences_id)
                tgt.append(json.loads(tgt_line.strip())[:len(cut_sentences_id)])

                assert len(src[-1]) == len(tgt[-1]), "The number of sentences is not equal to the labels!"

    with open('./data/process_src_ids.txt', 'w', encoding='utf-8') as file_src:
        with open('./data/process_labels_ids.txt', 'w', encoding='utf-8') as file_tgt:

            bar = tqdm(list(zip(src, tgt)), '读取原文文本和标签：')

            for src_id, tgt_id in bar:
                file_src.write(json.dumps(src_id) + '\n')
                file_tgt.write(json.dumps(tgt_id) + '\n')


if __name__ == "__main__":
    # process_src()
    # process_tgt()
    # process_src('test')
    # process_tgt('test')
    # greedy_generate_labels()
    tokenize_and_truncature()
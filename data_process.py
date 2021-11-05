
import json
from tqdm import tqdm
import config
import re
import numpy as np

from transformers import BertTokenizerFast


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set

def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences.
    """
    assert len(sentences) > 0
    assert n > 0

    # words = _split_into_words(sentences)

    words = sum(sentences, [])
    # words = [w for w in words if w not in stopwords]
    return _get_ngrams(n, words)

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

    if len(grams_model) == 0 or len(grams_reference) == 0:
        return 0

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


def process_train():
    """
    处理训练集，去除src或tgt空样例、src句子数少于min_src_nsent的样例
    :return:
    """

    src_list, tgt_list = [], []

    with open('./data/ssplit_train_src.json', 'r', encoding='utf-8') as file_s:
        with open('./data/ssplit_train_tgt.json', 'r', encoding='utf-8') as file_t:

            bar = tqdm(list(zip(file_s.readlines(), file_t.readlines())), '读取原文文本：')
            for line_s, line_t in bar:
                src = turn2doc(json.loads(line_s.strip()))
                tgt = turn2doc(json.loads(line_t.strip()))

                if src and tgt and len(src) > config.min_src_nsents:

                    src_list.append(src)
                    tgt_list.append(tgt)

    with open('./data/process_train_src.txt', 'w', encoding='utf-8') as file_s:
        with open('./data/process_train_tgt.txt', 'w', encoding='utf-8') as file_t:

            bar = tqdm(list(zip(src_list, tgt_list)), '读取原文文本：')
            for line_s, line_t in bar:
                file_s.write(json.dumps(line_s) + '\n')
                file_t.write(json.dumps(line_t) + '\n')


def process_test():
    src_list, tgt_list = [], []

    with open('./data/ssplit_test_src.json', 'r', encoding='utf-8') as file_s:
        with open('./data/ssplit_test_tgt.json', 'r', encoding='utf-8') as file_t:

            bar = tqdm(list(zip(file_s.readlines(), file_t.readlines())), '读取原文文本：')
            for line_s, line_t in bar:
                src = turn2doc(json.loads(line_s.strip()))
                tgt = turn2doc(json.loads(line_t.strip()))

                src_list.append(src)
                tgt_list.append(tgt)

    with open('./data/process_test_src.txt', 'w', encoding='utf-8') as file_s:
        with open('./data/process_test_tgt.txt', 'w', encoding='utf-8') as file_t:
            bar = tqdm(list(zip(src_list, tgt_list)), '读取原文文本：')
            for line_s, line_t in bar:
                file_s.write(json.dumps(line_s) + '\n')
                file_t.write(json.dumps(line_t) + '\n')


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0

    abstract = _rouge_clean(' '.join(abstract_sent_list)).split()
    sents = [_rouge_clean(s).split() for s in doc_sent_list]

    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))

            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i

        if cur_id == -1:
            return selected

        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def greedy_selection_all(doc_sent_list, abstract_sent_list):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    abstract = _rouge_clean(' '.join(abstract_sent_list)).split()
    sents = [_rouge_clean(s).split() for s in doc_sent_list]

    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    selected_1grams = []
    selected_2grams = []

    cur_max_rouge = 0.0

    for i in range(len(sents)):

        candidates_1 = selected_1grams + [evaluated_1grams[i]]
        candidates_1 = set.union(*map(set, candidates_1))

        candidates_2 = selected_2grams + [evaluated_2grams[i]]
        candidates_2 = set.union(*map(set, candidates_2))

        rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
        rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

        rouge_score = rouge_1 + rouge_2

        if rouge_score > cur_max_rouge:
            cur_max_rouge = rouge_score

            selected.append(i)
            selected_1grams.append(evaluated_1grams[i])
            selected_2grams.append(evaluated_2grams[i])

    if not selected:
        selected = [0, 1, 2]

    return sorted(selected)


# 获取oracle结果
def greedy_generate_oracle():

    # 读取原文和摘要
    with open('./data/process_test_src.txt', 'r', encoding='utf-8') as f_src:
        with open('./data/process_test_tgt.txt', 'r', encoding='utf-8') as f_tgt:
            with open('./data/oracle_cut.txt', 'w', encoding='utf-8') as f_pdt:

                bar = tqdm(list(zip(f_src.readlines(), f_tgt.readlines())), '生成oracle及标签')

                for j_src, j_tgt in bar:

                    src = json.loads(j_src.strip())
                    tgt = json.loads(j_tgt.strip())

                    cut_src = []
                    src_len = 0

                    for sent in src:

                        tokens = sent.split(' ')

                        # 去除过长的
                        if src_len + len(tokens) + 2 > config.max_position_embeddings:
                            break

                        # 去除多余句
                        if len(cut_src) + 1 > config.max_src_nsents:
                            break

                        cut_src.append(sent)
                        src_len += len(tokens) + 2

                    res = greedy_selection(cut_src, tgt, 3)

                    doc = []

                    for i in res:
                        doc.append(src[i])

                    f_pdt.write(' '.join(doc) + '\n')


def greedy_generate_labels():

    # 读取原文和摘要
    with open('./data/process_train_src.txt', 'r', encoding='utf-8') as f_src:
        with open('./data/process_train_tgt.txt', 'r', encoding='utf-8') as f_tgt:
            with open('./data/labels(v0.6)_n.txt', 'w', encoding='utf-8') as f_pdt:

                bar = tqdm(list(zip(f_src.readlines(), f_tgt.readlines())), '生成oracle及标签')

                count = 0

                for j_src, j_tgt in bar:

                    src = json.loads(j_src.strip())                 # 分句的形式
                    tgt = json.loads(j_tgt.strip())                 # 文档形式

                    cut_src = []
                    src_len = 0

                    for sent in src:

                        # 句子中的每个token
                        tokens = sent.split(' ')

                        # 去除过长的
                        if src_len + len(tokens) + 2 > config.max_position_embeddings:
                            break

                        # 去除多余句
                        if len(cut_src) + 1 > config.max_src_nsents:
                            break

                        cut_src.append(sent)
                        src_len += len(tokens) + 2

                    res = greedy_selection(cut_src, tgt, 3)

                    if not res:
                        count += 1

                    labels = [0] * len(src)     # 保持和原来一样的长度

                    for i in res:
                        labels[i] = 1

                    f_pdt.write(json.dumps(labels) + '\n')
                    # f_pdt.write(json.dumps(res) + '\n')

    print(count)


def true_num(labels):
    count = 0
    for i in labels:
        if i:
            count += 1
    return count


# 对数据提前进行tokenize和截断
def tokenize_and_truncature_old():

    tokenizer = BertTokenizerFast.from_pretrained('./bert-base-uncased')

    src, tgt = [], []

    with open('./data/process_train_src.txt', 'r', encoding='utf-8') as file_src:
        with open('./data/labels(v0.61).txt', 'r', encoding='utf-8') as file_tgt:

            bar = tqdm(list(zip(file_src.readlines(), file_tgt.readlines())), '读取原文文本和标签：')

            for src_line, tgt_line in bar:

                sentences = json.loads(src_line.strip())
                labels = json.loads(tgt_line.strip())

                # 对于其中的每一句都进行如下操作
                # 根据最大句子长度压缩每一个句子x
                # 根据最大长度压缩句子（512
                cut_sentences_id = []
                cut_labels = []
                article_len = 0

                first_true = -1
                last_true = len(labels)

                for i in range(len(labels)):
                    if labels[i]:
                        first_true = i if first_true == -1 else first_true
                        last_true = i

                head = 0
                tail = len(sentences)

                if len(sentences) > config.max_src_nsents:

                    left = first_true + 1
                    mid = last_true - first_true + 1
                    right = len(sentences) - last_true

                    if mid < config.min_src_nsents:
                        if left > right:
                            lam = max(0, config.min_src_nsents-(mid+right)) + 2
                            head = max(head, first_true - np.random.poisson(lam=lam))
                        else:
                            lam = max(0, config.min_src_nsents-(mid+left)) + 2
                            tail = min(tail, last_true + np.random.poisson(lam=lam))

                    else:
                        head = max(head, first_true - np.random.poisson(lam=2))
                        tail = min(tail, last_true + np.random.poisson(lam=2))

                for i in range(head, tail):

                    sentence = sentences[i]

                    # 截断句子长度
                    tokens = sentence.split(' ')
                    tail = min(config.max_sent_len, len(tokens))
                    sentence_id = tokenizer.encode(' '.join(['[CLS]'] + tokens[:tail] + ['[SEP]']), add_special_tokens=False)

                    # 防止总长度超过max_position_embedding
                    if article_len + len(sentence_id) > config.max_position_embeddings:
                        break

                    # 防止总句子数超过阈值
                    if len(cut_sentences_id) + 1 > config.max_src_nsents:
                        break

                    # 如果太短，就不考虑
                    if len(sentence_id) < config.min_src_ntokens_per_sent + 2:
                        continue

                    # 决定句子是否添加进当前样例
                    cut_labels.append(labels[i])
                    article_len += len(sentence_id)
                    cut_sentences_id.append(sentence_id)

                # 句子太少或者正例标签太少的样例都被剔除
                if len(cut_sentences_id) >= config.min_src_nsents and true_num(cut_labels) >= config.min_true_nlabels:

                    src.append(cut_sentences_id)
                    tgt.append(cut_labels)

                    assert len(src[-1]) == len(tgt[-1]), "The number of sentences is not equal to the labels!"

    with open('./data/process_src_ids(v0.61).txt', 'w', encoding='utf-8') as file_src:
        with open('./data/process_labels_ids(v0.61).txt', 'w', encoding='utf-8') as file_tgt:

            bar = tqdm(list(zip(src, tgt)), '写入原文文本和标签：')

            for src_id, tgt_id in bar:
                file_src.write(json.dumps(src_id) + '\n')
                file_tgt.write(json.dumps(tgt_id) + '\n')


def tokenize_and_truncature():

    tokenizer = BertTokenizerFast.from_pretrained('./bert-base-uncased')

    src, tgt = [], []

    with open('./data/process_train_src.txt', 'r', encoding='utf-8') as file_src:
        with open('./data/labels(v0.6)_n.txt', 'r', encoding='utf-8') as file_tgt:

            bar = tqdm(list(zip(file_src.readlines(), file_tgt.readlines())), '读取原文文本和标签：')

            for src_line, tgt_line in bar:

                sentences = json.loads(src_line.strip())
                labels = json.loads(tgt_line.strip())

                # 对于其中的每一句都进行如下操作
                # 根据最大句子长度压缩每一个句子x
                # 根据最大长度压缩句子（512
                cut_sentences_id = []
                cut_labels = []
                article_len = 0

                for i in range(len(sentences)):

                    sentence = sentences[i]

                    # 截断句子长度
                    tokens = sentence.split(' ')
                    sentence_id = tokenizer.encode(' '.join(['[CLS]'] + tokens + ['[SEP]']), add_special_tokens=False)

                    # 防止总长度超过max_position_embedding
                    if article_len + len(sentence_id) > config.max_position_embeddings:
                        break

                    # 防止总句子数超过阈值
                    if len(cut_sentences_id) + 1 > config.max_src_nsents:
                        break

                    # 如果太短，就不考虑
                    if len(sentence_id) < config.min_src_ntokens_per_sent + 2:
                        continue

                    # 决定句子是否添加进当前样例
                    cut_labels.append(labels[i])
                    article_len += len(sentence_id)
                    cut_sentences_id.append(sentence_id)

                # 句子太少或者正例标签太少的样例都被剔除
                if len(cut_sentences_id) >= config.min_src_nsents and true_num(cut_labels) >= config.min_true_nlabels:

                    src.append(cut_sentences_id)
                    tgt.append(cut_labels)

                    assert len(src[-1]) == len(tgt[-1]), "The number of sentences is not equal to the labels!"

    with open('./data/process_src_ids(v0.6)_n.txt', 'w', encoding='utf-8') as file_src:
        with open('./data/process_labels_ids(v0.6)_n.txt', 'w', encoding='utf-8') as file_tgt:

            bar = tqdm(list(zip(src, tgt)), '写入原文文本和标签：')

            for src_id, tgt_id in bar:
                file_src.write(json.dumps(src_id) + '\n')
                file_tgt.write(json.dumps(tgt_id) + '\n')


def tokenize_and_truncature_for_test():

    tokenizer = BertTokenizerFast.from_pretrained('./bert-base-uncased')

    src = []

    count = 0

    with open('./data/process_test_src.txt', 'r', encoding='utf-8') as file_src:

            bar = tqdm(file_src.readlines(), '读取测试文本：')
            for src_line in bar:

                sentences = json.loads(src_line.strip())
                cut_sentences_id = []

                # 对于其中的每一句都进行如下操作
                # 根据最大句子长度压缩每一个句子x
                # 根据最大长度压缩句子（512

                article_len = 0

                for i in range(len(sentences)):

                    sentence = sentences[i]

                    # 截断句子长度
                    tokens = sentence.split()

                    sentence_id = tokenizer.encode(' '.join(['[CLS]'] + tokens + ['[SEP]']),
                                                   add_special_tokens=False)

                    # 防止总长度超过max_position_embedding
                    if article_len + len(sentence_id) > config.max_position_embeddings:
                        break

                    # 防止总句子数超过阈值
                    if len(cut_sentences_id) + 1 > config.max_src_nsents:
                        break

                    # 如果太短，就不考虑
                    if len(sentence_id) < config.min_src_ntokens_per_sent + 2:
                        continue

                    article_len += len(sentence_id)
                    cut_sentences_id.append(sentence_id)

                if not cut_sentences_id:
                    count += 1

                src.append(cut_sentences_id)

    print(count)

    with open('./data/process_src_ids_test.txt', 'w', encoding='utf-8') as file_src:

        bar = tqdm(src, '写入测试文本ids：')

        for src_id in bar:
            file_src.write(json.dumps(src_id) + '\n')


# 根据序号抽取句子
def get_candidate_summary():

    tokenizer = BertTokenizerFast.from_pretrained('./bert-base-uncased')
    count = 0

    with open('./data/process_src_ids_test.txt', 'r', encoding='utf-8') as file_src:
        with open('./res_sent_indices_v12_1.txt', 'r', encoding='utf-8') as file_can:
            with open('./res_sent_values_v12_1.txt', 'r', encoding='utf-8') as file_val:
                with open('./system_extbert_v12_1.txt', 'w', encoding='utf-8') as file_system:

                    bar = tqdm(list(zip(file_src.readlines(), file_can.readlines(), file_val.readlines())), '读取测试文本：')
                    for src_line, sent_indices, sent_values in bar:

                        src_ids_list = json.loads(src_line.strip())
                        sent_indices_list = json.loads(sent_indices.strip())
                        sent_values_list = json.loads(sent_values.strip())

                        text = []

                        # new_sent_indices_list = []
                        #
                        # for sent_idx, sent_val in zip(sent_indices_list, sent_values_list):
                        #     if sent_val > 0.5:
                        #         new_sent_indices_list.append(sent_idx)
                        #
                        # if not new_sent_indices_list:
                        #     new_sent_indices_list = sent_indices_list

                        new_sent_indices_list = sent_indices_list
                        new_sent_indices_list = list(sorted(new_sent_indices_list))

                        if not new_sent_indices_list:
                            count += 1

                        for sent_idx in new_sent_indices_list:

                            if sent_idx >= len(src_ids_list):
                                continue
                            else:
                                sentence = tokenizer.decode(src_ids_list[sent_idx][1:-1])
                                text.append(sentence)

                        file_system.write(' '.join(text) + '\n')
    print(count)


# 只抽前3句
def lead_3():

    with open('./data/process_test_src.txt', 'r', encoding='utf-8') as file_src:
        with open('./system_top3.txt', 'w', encoding='utf-8') as file_system:

            bar = tqdm(file_src.readlines(), '读取测试文本：')
            for src_line in bar:

                src_list = json.loads(src_line.strip())

                text = []

                for i in range(3):
                    if i >= len(src_list):
                        break
                    text.append(src_list[i])

                file_system.write(' '.join(text) + '\n')


# 比较两种处理数据的方式
def compare_dataset():

    with open('./data/process_labels_ids(new).txt', 'r', encoding='utf8') as file1:

        count = {}

        for line in file1.readlines():

            labels = json.loads(line.strip())

            sent_num = 0

            for j in range(len(labels)):
                if labels[j] == 1:
                    sent_num += 1

            if sent_num in count:
                count[sent_num] += 1
            else:
                count[sent_num] = 1

        item = count.items()
        item = dict(sorted(item, key=lambda x:x[1], reverse=True))
        print(item)



if __name__ == "__main__":
    # process_train()
    # process_test()

    # greedy_generate_oracle()
    # greedy_generate_labels()
    # tokenize_and_truncature()

    # tokenize_and_truncature_for_test()

    get_candidate_summary()
    # get_two_candidate_summary()
    # lead_3()
    # compare_dataset()
    # update_system_labels()

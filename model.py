from transformers import BertModel, BertTokenizer, BertConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

import math
import config


from transformer_module import (
    MultiHeadedAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    TransformerEncoderLayer,
    Classifier
)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """
        :param top_vecs: [batch, sent_num, hidden_size]
        :param mask: [batch, sent_num]
        :return: [batch, sent_num]
        """

        batch_size, n_sents = top_vecs.size()[:2]
        pos_emb = self.pos_emb.pe[:, :n_sents]      # [1, sent_num, hidden_size]

        if pos_emb.size()[1:] != top_vecs.size()[1:]:
            print(batch_size, n_sents)
            print(pos_emb.size(), top_vecs.size())

        x = top_vecs + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1)

        return sent_scores

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if large:
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, input_ids, attention_mask, token_type_ids):

        if self.finetune:
            top_vec, _ = self.model(input_ids, attention_mask, token_type_ids)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(input_ids, attention_mask, token_type_ids)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, checkpoint=None):
        super(ExtSummarizer, self).__init__()

        self.bert = BertModel.from_pretrained('./bert-base-uncased')

        self.ext_layer = ExtTransformerEncoder(config.hidden_size, config.ext_ff_size, config.ext_heads,
                                               config.ext_dropout, config.ext_layers)

        self.loss_func = nn.BCELoss(reduction='none')

        if config.model_type == 'baseline':
            bert_config = BertConfig()
            self.bert = BertModel(bert_config)
            self.ext_layer = Classifier(config.hidden_size)

        if config.max_position_embeddings > 512:
            my_pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.embeddings.position_embeddings.weight.data[-1][None,:].repeat(config.max_position_embeddings-512, 1)
            self.bert.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if config.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-config.param_init, config.param_init)
            if config.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

    def forward(self, src, segs, clss, mask_src, mask_cls, labels=None):
        """
        :param src: 原文, [batch, src_len]
        :param segs: 句子标签, [batch, src_len]
        :param clss: 每个cls标签的位置, 是一个列表，[[cls1,cls2,...],...,[]]
        :param mask_src: 原文的mask，[batch, src_len]
        :param mask_cls: cls标签的mask（句子序列的mask）, [batch, sent_num]
        :param labels: 句子标签, [bacth, sent_num]
        :return:
        """

        # 得到每一个token的编码
        outputs = self.bert(input_ids=src, attention_mask=mask_src, token_type_ids=segs)

        top_vec = outputs.last_hidden_state     # [batch, src_len, hidden_size]

        # 取出每个[CLS]的编码作为每个句子的编码
        # clss为每一个CLS的索引
        batch_size, _, hidden_size = top_vec.size()

        sent_num = [len(d) for d in clss]
        max_sent_num = max(sent_num)

        sents_vec = torch.zeros((batch_size, max_sent_num, hidden_size)).to(config.device)

        # sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        # sents_vec = sents_vec * mask_cls[:, :, None].float()

        for i in range(batch_size):
            for j in range(len(clss[i])):
                sents_vec[i, j] = top_vec[i, clss[i][j]]

        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)

        if labels is not None:
            loss = self.loss_func(sent_scores, labels)
            loss = torch.sum(loss * mask_cls) / torch.sum(mask_cls)
            return loss
        else:
            return sent_scores


if __name__ == "__main__":

    model = ExtSummarizer()




from transformers import AdamW, BertTokenizerFast, get_linear_schedule_with_warmup
from model import ExtSummarizer

import os
import numpy as np

import time

import torch
import torch.nn as nn

import config
import argparse


from data_utils import get_dataloader

import json
from tqdm import tqdm

random_seed = 666
torch.manual_seed(random_seed)

class Trainer:

    def __init__(self):

        self.tokenizer = BertTokenizerFast.from_pretrained('./bert-base-uncased')

    def set_model(self, checkpoint_path=None):

        checkpoint = None

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)

        self.model = ExtSummarizer(checkpoint).to(config.device)
        # self.model = EncoderDecoderModel(config=Trans_config)
        # Initializing a Bert2Bert model from the bert-base-uncased style configurations
        # self.model = TransGraph(Seq2SeqModel.encoder, Seq2SeqModel.decoder).to(self.device)

        # self.model = nn.parallel.DataParallel(self.model, device_ids=[0]).to(self.device)
        # self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def show_parameters(self):
        # 定义总参数量、可训练参数量及非可训练参数量变量
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0

        # 遍历model.parameters()返回的全局参数列表
        for param in self.model.parameters():
            mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

    def list2tensor(self, src, tgt):
        """
        该函数主要处理以下问题：
        1. 将src进行tokenize
        2. 填充src和tgt到batch内最大句子数
        :param src:原文，每一篇文档以句子形式划分，是ids的形式[[sent1],[sent2],...]
        :param tgt:标签，对应每一句话，0为不抽取，1为抽取[0,1,0,0,1,0,1]
        :return:inputs，inputs_mask，segments，labels
        """

        # 防止不够config.batch_size的情况发生
        batch_size = len(src)

        # 获取src_sent_lens(每句话的长度),tgt_lens(每篇文档的句子数),src_lens(每篇文档的总长度)
        src_sent_lens, tgt_lens = [[len(s) for s in src[b]] for b in range(batch_size)], [len(t) for t in tgt]
        src_lens = [sum(src_sent_lens[b]) for b in range(batch_size)]

        # 得到一个batch内的最大文章长度和最大句子个数
        src_max_len = max(src_lens)
        tgt_max_len = max(tgt_lens)

        # 输入编码，标签编码，句子编码，输入mask
        inputs = torch.zeros((batch_size, src_max_len), dtype=torch.int64)

        labels = torch.zeros((batch_size, tgt_max_len))

        segments = torch.zeros((batch_size, src_max_len), dtype=torch.int64)
        inputs_mask = torch.zeros((batch_size, src_max_len), dtype=torch.bool)

        cls_indices = []
        cls_mask = torch.zeros((batch_size, tgt_max_len), dtype=torch.bool)

        # 对于每一个batch
        for i in range(batch_size):

            # 句子头指针
            seg_h = 0
            next_embed = 0
            cls_index = []

            # 对于第i个batch中的每一句话
            for j in range(len(src[i])):

                inputs[i, seg_h:seg_h+src_sent_lens[i][j]] = torch.LongTensor(src[i][j])
                segments[i, seg_h:seg_h+src_sent_lens[i][j]] = next_embed
                cls_index.append(seg_h)

                seg_h = seg_h + src_sent_lens[i][j]
                next_embed = 1 - next_embed

            cls_indices.append(cls_index)

            # 填充的部分
            segments[i, seg_h:] = next_embed

            inputs_mask[i, :src_lens[i]] = True
            cls_mask[:tgt_lens[i]] = True

            labels[i, :tgt_lens[i]] = torch.Tensor(tgt[i])

        return inputs, inputs_mask, segments, labels, cls_indices, cls_mask

    def list2tensor_test(self, src):

        # 防止不够config.batch_size的情况发生
        # 防止不够config.batch_size的情况发生
        batch_size = len(src)

        # 获取src_sent_lens(每句话的长度),src_sent_num(每篇文档的句子数),src_lens(每篇文档的总长度)
        src_sent_lens = [[len(s) for s in src[b]] for b in range(batch_size)]
        src_sent_num = [len(d) for d in src_sent_lens]
        src_lens = [sum(src_sent_lens[b]) for b in range(batch_size)]

        # 得到一个batch内的最大文章长度和最大句子个数
        src_max_len = max(src_lens)
        max_sent_num = max(src_sent_num)

        # 输入编码，标签编码，句子编码，输入mask
        inputs = torch.zeros((batch_size, src_max_len), dtype=torch.int64)

        segments = torch.zeros((batch_size, src_max_len), dtype=torch.int64)
        inputs_mask = torch.zeros((batch_size, src_max_len), dtype=torch.bool)

        cls_indices = []
        cls_mask = torch.zeros((batch_size, max_sent_num), dtype=torch.bool)

        # 对于每一个batch
        for i in range(batch_size):

            # 句子头指针
            seg_h = 0
            next_embed = 0
            cls_index = []

            # 对于第i个batch中的每一句话
            for j in range(len(src[i])):
                inputs[i, seg_h:seg_h + src_sent_lens[i][j]] = torch.LongTensor(src[i][j])
                segments[i, seg_h:seg_h + src_sent_lens[i][j]] = next_embed
                cls_index.append(seg_h)

                seg_h = seg_h + src_sent_lens[i][j]
                next_embed = 1 - next_embed

            cls_indices.append(cls_index)

            # 填充的部分
            segments[i, seg_h:] = next_embed

            inputs_mask[i, :src_lens[i]] = True
            cls_mask[:src_sent_num[i]] = True

        return inputs, inputs_mask, segments, cls_indices, cls_mask

    def train(self, vision='v1'):
        """
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        trainer.train()
        """
        src_path = './data/process_src_ids.txt'
        tgt_path = './data/process_labels_ids.txt'

        self.train_loader = get_dataloader(src_path, tgt_path, batch_size=config.batch_size)  # 处理成多个batch的形式

        optimizer = AdamW(self.model.parameters(), lr=config.lr, betas=(0.9, 0.998), weight_decay=1e-3)

        total_steps = len(self.train_loader) * config.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=8000, num_training_steps=total_steps)

        for epoch in range(config.epochs):
            self.model.train()

            # 训练
            bar = tqdm(range(len(self.train_loader)),
                       desc="epoch:%s, batch:%s, step:%s，Lc:None" % (epoch + 1, 1, epoch * len(self.train_loader)), ncols=150)

            for step, batch in zip(bar, self.train_loader):
                optimizer.zero_grad()

                inputs, inputs_mask, segments, labels, cls_indices, cls_mask = self.list2tensor(*batch)

                if step < 200:
                    continue

                loss = self.model(inputs.to(config.device),
                                  segments.to(config.device),
                                  cls_indices,
                                  inputs_mask.to(config.device),
                                  cls_mask.to(config.device),
                                  labels.to(config.device))

                loss = loss.mean()
                loss.backward()

                记录四个loss
                with open('loss_c_%s.txt' % vision, 'a+', encoding="UTF-8") as file:
                    file.write(str(loss.item()) + '\n')

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                bar.set_description("epoch:%s, batch:%s, step:%s，Lc:%.5f" % (epoch + 1, step, epoch * len(self.train_loader) + step, loss.item()))

            if epoch >= 0:
                state = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch + 1
                }
                torch.save(state, './model_save/checkpoint_%s_%s' % (vision, epoch + 1))

    def eval_model(self):
        self.model.eval()

        src_path = './data/ssplit_test_src.json'

        self.test_loader = get_dataloader(src_path, None, self.tokenizer, shuffle=False, batch_size=16)  # 处理成多个batch的形式

        with torch.no_grad():
            bar = tqdm(range(len(self.test_loader)))
            for step, batch in zip(bar, self.test_loader):

                inputs, inputs_mask, segments, cls_indices, cls_mask = self.list2tensor_test(*batch)

                logits = self.model(inputs.to(config.device),
                                    segments.to(config.device),
                                    cls_indices,
                                    inputs_mask.to(config.device),
                                    cls_mask.to(config.device))

                # 输出前k句的
                output_sentences_indices = logits.argmax(dim=-1)[:, :config.topk].tolist()

                with open('./res_sent_indices.txt', 'a', encoding='utf-8') as file:
                    for ids in output_sentences_indices:
                        file.write(json.dumps(ids)+'\n')



def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./model_save/checkpoint_v1_1')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_config()

    model = Trainer()

    if args.train:
        model.set_model()
        model.show_parameters()
        model.train()

    if args.eval and args.model_path:
        model.set_model(args.model_path)
        model.show_parameters()
        model.eval_model()

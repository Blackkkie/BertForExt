
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_type = "ExtBert"

topk = 3

batch_size = 16
lr = 1e-3
epochs = 100
max_sent_len = 50
max_src_nsents = 20

# Bert配置
attention_probs_dropout_prob = 0.1
hidden_dropout_prob = 0.1
hidden_size = 768
initializer_range = 0.02
intermediate_size = 3072
layer_norm_eps = 1e-12
max_position_embeddings = 512

num_attention_heads = 12
num_hidden_layers = 12
pad_token_id = 0
position_embedding_type = "absolute"

type_vocab_size = 2
use_cache = True
vocab_size = 30522


# 句子分类器配置
param_init = 0.1
param_init_glorot = True

ext_dropout = 0.2
ext_layers = 2
ext_hidden_size = 768
ext_heads = 8
ext_ff_size = 2048


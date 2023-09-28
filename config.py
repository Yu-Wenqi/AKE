# -*- coding: utf-8 -*-
# @Time    : 2022/5/2 11:25
# @Author  : leizhao150
import torch

tag2ids = {'[PAD]': 0,'B': 1, 'I': 2, 'E': 3,'S': 4,"O": 5}
id2tags = {val: key for key, val in tag2ids.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = './datas/320/json/train1.json'  # 1-5
test_path = './datas/320/json/test1.json'
vocab_path = './datas/320/json/vocab.json'
# train_path = './datas/5190/json/train.json'
# test_path = './datas/5190/json/test.json'
# vocab_path = './datas/5190/json/vocab.json'
logfile='./result/log/320/BiLSTM/1/train1-1.txt'
save_path = './result/casestudy/320/BiLSTM/1/train1-1.txt'
feature = "no feature" #no feature/FFD/FN/TFD/FFD+FN/FN+TFD/FFD+TFD/FFD+FN+TFD
# "E:\MyPythonProjects\eyemovements\result\prediction\little\BiLSTM"


# 训练参数(att-320)
fs_num = 0
embed_dim = 64     # 64
hidden_dim = 128    # 512
batch_size = 32    # 64
max_length = 512  #句子最大长度（2的次方） 64
#词量
vocab_size = 1226   #1226和  2121/2138
dropout_value = 0.5
emb_dropout_value = 0.5
lstm_dropout_value = 0.2
linear_dropout_value = 0.2

lr = 0.005   # 0.003
layers_num = 1
weight_decay = 1e-6   #1e-6
factor = 0.5
patience = 3
epochs = 100

# # 训练参数(att-5190)
# fs_num = 2
# embed_dim = 64     # 64
# hidden_dim = 128    # 512
# batch_size = 32    # 64
# max_length = 512  #句子最大长度（2的次方） 64
# #词量
# vocab_size = 2119   #1226和  2121/2138
# dropout_value = 0.5
# emb_dropout_value = 0.5
# lstm_dropout_value = 0.2
# linear_dropout_value = 0.2
#
# lr = 0.003   # 0.003
# layers_num = 1
# weight_decay = 1e-6   #1e-6
# factor = 0.5
# patience = 3
# epochs = 30

# # BiLSTM/BiLSTM-CRF 5190 训练参数
# fs_num = 2
# embed_dim = 64     # 64
# hidden_dim = 128    # 512
# batch_size = 32    # 64
# max_length = 512  #句子最大长度（2的次方） 64
# #词量
# vocab_size = 2119   #1226和  2121/2138
# dropout_value = 0.5
# emb_dropout_value = 0.5
# lstm_dropout_value = 0.2
# linear_dropout_value = 0.2
#
# lr = 0.003   # 0.003
# layers_num = 1
# weight_decay = 1e-6   #1e-6
# factor = 0.5
# patience = 3
# epochs = 30


# 训练参数：BiLSTM/BiLSTM-CRF-320
# fs_num =0
# embed_dim = 64     # 64
# hidden_dim = 128    # 512
# batch_size = 32    # 64
# max_length = 512  #句子最大长度（2的次方） 64
# #词量
# vocab_size = 1226   #1226和  2121/2138
# dropout_value = 0.5
# emb_dropout_value = 0.5
# lstm_dropout_value = 0.2
# linear_dropout_value = 0.2
#
# lr = 0.01   # 0.003
# layers_num = 1
# weight_decay = 1e-6   #1e-6
# factor = 0.5
# patience = 3
# epochs = 30
#

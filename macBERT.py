from tqdm import tqdm
import numpy as np
import logging
from tqdm import tqdm
from config import *
import torch.nn as nn
import torch
from evaluate import *
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast


weight = "hfl/chinese-macbert-base"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
max_length = 500


# Load the train data
def load_traindata(train_path):
    train_file = json.load(open(train_path,'r',encoding='utf-8'))
    train_sens, train_features, train_tags = [],[],[]
    train_word_nums = []

    sens = ''
    nums = 0
    for key in train_file.keys():
        tags = []
        items = train_file[key]
        sens = ''
        nums = 0
        features = []
        for item in items:
            sens += item[0]
            sens += ' '
            tags.append(item[-1])
            features.append(item[1:-1])
            nums += 1
        train_sens.append(sens.strip())
        train_word_nums.append(nums)
        train_tags.append(tags)
        train_features.append(features)
    return train_sens, train_word_nums, train_tags, train_features

# Load the test data
def load_testdata(test_path):
    test_file = json.load(open(test_path, 'r', encoding='utf-8'))
    test_sens, test_features, test_tags = [],[],[]
    test_word_nums = []

    sens = ''
    nums = 0
    for key in test_file.keys():
        tags = []
        items = test_file[key]
        sens = ''
        nums = 0
        features = []
        for item in items:
            sens += item[0]
            sens += ' '
            tags.append(item[-1])
            features.append(item[1:-1])
            nums += 1
        test_sens.append(sens.strip())
        test_word_nums.append(nums)
        test_tags.append(tags)
        test_features.append(features)
    return test_sens, test_word_nums, test_tags, test_features


def align_label(text, labels, features):
    input = tokenizer(text, max_length=max_length, add_special_tokens=True, padding='max_length', truncation=True,
                      return_tensors='pt')
    word_ids = input.word_ids()
    input_ids = input['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    previous_word_idx = None
    new_labels = []
    new_features = []
    no_features = [0 for i in range(1, 4)]

    for word_idx in word_ids:
        if word_idx is None:
            new_labels.append('none')
            new_features.append(no_features)
        #   new_labels.append('O')

        elif word_idx != previous_word_idx:
            try:
                new_labels.append(labels[word_idx])
                new_features.append(features[word_idx])
            except:
                new_labels.append('none')
                new_features.append(no_features)
            #   new_labels.append('O')
        else:
            try:
                new_labels.append(labels[word_idx] if label_all_tokens else 'none')
                new_features.append(features[word_idx] if label_all_tokens else no_features)
            #   new_labels.append(labels[word_idx] if label_all_tokens else 'O')
            except:
                new_labels.append('none')
                new_features.append(no_features)
        previous_word_idx = word_idx

    label_ids = [tag2ids[label] for label in new_labels]

    return label_ids, tokens, new_features


class MyDataset(Dataset):
    def __init__(self, texts, old_features, tags):
        self.texts = texts
        self.tags = tags

        self.old_features = old_features

        self.labels = []
        self.tokens = []
        self.features = []

        self.input_ids = None
        self.attention_masks = None

    def encode(self):
        for i in tqdm(range(len(self.texts))):
            text = self.texts[i]
            tag = self.tags[i]
            feature = self.old_features[i]
            tags, tokens, features = align_label(text, tag, feature)
            self.labels.append(tags)
            self.tokens.append(tokens)
            self.features.append(features)

        self.features = np.array(self.features, float)
        self.inputs = tokenizer(self.texts, max_length=max_length, add_special_tokens=True, padding='max_length',
                                truncation=True, return_tensors='pt')
        self.input_ids = self.inputs['input_ids']
        self.attention_masks = self.inputs['attention_mask']

    def __getitem__(self, idx):
        return self.input_ids[idx, :], self.attention_masks[idx, :], self.tokens[idx], torch.tensor(self.features[idx],
                                                                                                    dtype=torch.float32), torch.tensor(
            self.labels[idx])

    def __len__(self):
        return len(self.input_ids)


class BertNerModel(nn.Module):
    def __init__(self, num_labels):
        super(BertNerModel, self).__init__()

        self.bert = BertModel.from_pretrained(weight)
        self.dropout = nn.Dropout(0.1)

        # 特征数
        self.classifier = nn.Linear(768 + 2, num_labels)

    def forward(self, input_ids, attention_mask, extra_features, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_output = outputs[0]
        bert_outputs = self.dropout(pooled_output)

        # 特征
        outputs = torch.concat((bert_outputs, extra_features[:, :, :]), -1)

        # 13特征
        # outputs = torch.concat((bert_outputs,extra_features[:,:,::2]),-1)

        # 无特征
        # outputs = bert_outputs
        outputs = self.classifier(outputs)

        return outputs

#修改pos转化为词-函数
def TagConvert(raw_tags, words_set, poss=None):

    true_tags = []
    for i in range(raw_tags.shape[0]):
        kw_list = []
        nkw_list = ""
        for j in range(len(raw_tags[i])):
            item = raw_tags[i][j]
            if item == 0:
                continue
            if poss !=None and j in poss[i]:
                continue
            if item == 5:
                # continue
                nkw_list = ""
            if item == 4:
                if nkw_list not in kw_list:
                    kw_list.append(str(words_set[j][i]))
            if item == 1:
                nkw_list += str(words_set[j][i])
            if item == 2:
                nkw_list += str(words_set[j][i])
            if item == 3:
                nkw_list += str(words_set[j][i])
                if nkw_list not in kw_list:
                    kw_list.append(nkw_list)
                nkw_list = ""

        true_tags.append(kw_list)
    return true_tags

def macBERT():

    # Load the data
    train_sens, train_word_nums, train_tags, train_features = load_traindata(train_path)
    test_sens, test_word_nums, test_tags, test_features = load_traindata(test_path)

    # Define the model architecture and load the weights
    tokenizer = BertTokenizerFast.from_pretrained(weight)

    # Biuld the train data set
    train_dataset = MyDataset(train_sens, train_features, train_tags, tokenizer)
    train_dataset.encode()

    # Biuld the test data set
    test_dataset = MyDataset(test_sens, test_features, test_tags)
    test_dataset.encode()

    # Biuld the data loader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=128)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=128)

    # Start training
    model = BertNerModel(num_labels=6)
    model = model.to(device)

    # Define the optimizer
    optim = AdamW(model.parameters(),lr=5e-5,weight_decay=1e-2)
    loss_fn = CrossEntropyLoss(reduction='none', ignore_index=0)
    loss_fn = loss_fn.to(device)

    epoch3, epoch5, epoch10, best_P3, best_R3, best_F3, best_P5, best_R5, best_F5, best_P10, best_R10, best_F10 = 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    fin_targets = []
    fin_prediction = []
    for epoch in tqdm(range(epochs)):
        loss_value = 0.0
        model.train()
        label_true, label_pred = [], []
        for i, batch in enumerate(train_dataloader):
            optim.zero_grad()
            input_ids, attention_masks, _, features, tags = batch
            pred_tags = model(input_ids.to(device), attention_masks.to(device), features.to(device))

            loss = loss_fn(pred_tags.permute(0, 2, 1), tags.to(device))
            loss = loss.mean()
            loss.backward()
            optim.step()

            pred_tags = F.softmax(pred_tags, dim=-1)
            pred_tags = torch.argmax(pred_tags, dim=-1)

            y_true = pred_tags.to(device).tolist()
            y_pred = tags.to(device).tolist()

            label_true.extend(y_true)
            label_pred.extend(y_pred)

            loss_value += loss.item()
        print("avg_loss: %.2f" % np.average(loss_value))
        logging.info("epoch: " + str(epoch))
        logging.info("avg_loss: %.2f" % np.average(loss_value))


        model.eval()
        kw_true, kw_pred = [], []
        label_true, label_pred = [], []
        for i, batch in enumerate(test_dataloader):
            input_ids, attention_masks, tokens, features, tags = batch
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = 0
                        module.train(False)
                pred_tags = model(input_ids.to(device), attention_masks.to(device), features.to(device))
                pred_tags = F.softmax(pred_tags, dim=-1)
                pred_tags = torch.argmax(pred_tags, dim=-1)

            #   y_pred, y_true = calculate_f1(pred_tags, tags)
            y_true = pred_tags.to(device).tolist()
            y_pred = tags.to(device).tolist()

            label_true.extend(y_true)
            label_pred.extend(y_pred)

            # more balance evaluate
            poss = []
            for i in range(len(tags)):
                pos = []
                for j in range(len(tags[i])):
                    if tags[i][j] == 0:
                        pos.append(j)
                poss.append(pos)

            kw_true.extend(TagConvert(tags, tokens))
            kw_pred.extend(TagConvert(pred_tags, tokens, poss))
        print(kw_pred)
        print(kw_true)

        # label_f1 = f1_score(label_true, label_pred, average='macro')
        P3, R3, F3 = evaluate3(kw_true, kw_pred)
        P5, R5, F5 = evaluate5(kw_true, kw_pred)
        P10, R10, F10 = evaluate10(kw_true, kw_pred)

        # torch.save(model.state_dict(),'./models/bert_5190_123_3.pt')
        # torch.save(model.state_dict(),'./models/bert/bert_320_train1_23_10.pt')
        if F3 > best_F3:
            best_F3 = F3
            best_P3 = P3
            best_R3 = R3
            epoch3 = epoch
            fin_targets = kw_pred
            fin_prediction = kw_true
        if F5 > best_F5:
            best_F5 = F5
            best_P5 = P5
            best_R5 = R5
            epoch5 = epoch
            fin_targets = kw_pred
            fin_prediction = kw_true
        if F10 > best_F10:
            best_F10 = F10
            best_P10 = P10
            best_R10 = R10
            epoch10 = epoch

    print(len(fin_targets))
    print(len(fin_prediction))

    # 将预测结果和目标结果存到txt中
    with open(save_path, mode='a+', encoding='utf-8') as f:
        len1 = len(fin_prediction)
        for i in range(0, len1):
            num1 = len(fin_prediction[i])
            st1 = ''
            for j in range(0, num1):
                word1 = fin_prediction[i][j]
                # print(word1)
                st1 = st1 + word1 + ','
            f.write(st1)
            f.write('\n')
        f.write('----------------------')
        f.write('\n')
        for i in range(0, len1):
            num2 = len(fin_targets[i])
            st2 = ''
            for j in range(0, num2):
                word2 = fin_targets[i][j]
                # print(word2)
                st2 = st2 + word2 + ','
            f.write(st2)
            f.write('\n')

    return epoch3, epoch5, epoch10, best_P3, best_R3, best_F3, best_P5, best_R5, best_F5, best_P10, best_R10, best_F10
    # return best_P, best_R, best_F, best_10F, best_15F

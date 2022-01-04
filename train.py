# -*- coding: utf-8 -*-

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import transformers
import argparse

from models.bert_CNN import bert_cnn, bert_cnn_Config
from models.bert_lr import bert_lr, bert_lr_Config
from models.bert_lr_last4layer import bert_lr_last4layer, bert_lr_last4layer_Config

from tools.config import Config
from transformers import AdamW
from transformers import BertConfig, BertForSequenceClassification
from tools.utils import SentimentDataset, convert_text_to_ids, seq_padding


class transformers_bert_binary_classification(object):
    def __init__(self,args):
        self.config = Config()
        self.device_setup(args)

    def device_setup(self,args):
        """
        设备配置并加载BERT模型
        :return:
        """
        self.freezeSeed()
        # 使用GPU，通过model.to(device)的方式使用
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        import os
        result_dir = args.result_dir
        if len(os.listdir(result_dir))  < 3:
            # 因为有一个trained文件，重新训练
            MODEL_PATH = self.config.get("BERT_path", "file_path")
            config_PATH = self.config.get("BERT_path", "config_path")
            vocab_PATH = self.config.get("BERT_path", "vocab_path")
        else:
            # 读取之前训练好的模型
            MODEL_PATH = self.config.get("result", "model_save_path")
            config_PATH = self.config.get("result", "config_save_path")
            vocab_PATH = self.config.get("result", "vocab_save_path")

        num_labels = self.config.get("training_rule", "num_labels")
        hidden_dropout_prob = self.config.get("training_rule", "hidden_dropout_prob")

        # 通过词典导入分词器
        self.tokenizer = transformers.BertTokenizer.from_pretrained(vocab_PATH)
        self.model_config = BertConfig.from_pretrained(config_PATH, num_labels=num_labels,
                                                       hidden_dropout_prob=hidden_dropout_prob)
        """
        train loss:  0.10704718510208534 	 train acc: 0.9637151849872321
        valid loss:  0.17820182011222863 	 valid acc: 0.9459971577451445
        """
        if args.my_model == "huggingface_default":
            self.model = BertForSequenceClassification.from_pretrained(MODEL_PATH, config=self.model_config)
        elif args.my_model == "bert_CNN":
            self.model = bert_cnn(bert_cnn_Config())
        elif args.my_model == "bert_lr":
            self.model = bert_lr(bert_lr_Config()) # 实现官方的BertForSequenceClassification
        elif args.my_model == "bert_lr_last4layer":
            self.model = bert_lr_last4layer(bert_lr_last4layer_Config())
        else:
            raise ValueError("无改模型")

        self.model.to(self.device)



    def model_setup(self):
        weight_decay = self.config.get("training_rule", "weight_decay")
        learning_rate = self.config.get("training_rule", "learning_rate")

        # 定义优化器和损失函数
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def get_data(self):
        """
        读取数据
        :return:
        """
        train_set_path = self.config.get("data_path", "trainingSet_path")
        valid_set_path = self.config.get("data_path", "valSet_path")
        test_set_path = self.config.get("data_path", "testingSet_path")

        batch_size = self.config.get("training_rule", "batch_size")

        # 数据读入
        # 加载数据集
        sentiment_train_set = SentimentDataset(train_set_path)
        sentiment_train_loader = DataLoader(sentiment_train_set, batch_size=batch_size, shuffle=True, num_workers=3)
        sentiment_valid_set = SentimentDataset(valid_set_path)
        sentiment_valid_loader = DataLoader(sentiment_valid_set, batch_size=batch_size, shuffle=False, num_workers=3)
        sentiment_test_set = SentimentDataset(test_set_path)
        sentiment_test_loader = DataLoader(sentiment_test_set, batch_size=batch_size, shuffle=False, num_workers=3)


        return sentiment_train_loader, sentiment_valid_loader,sentiment_test_loader

    def train_an_epoch(self, iterator):
        self.model_setup()
        epoch_loss = 0
        epoch_acc = 0

        for i, batch in enumerate(iterator):
            label = batch["label"]
            text = batch["text"]
            # 使用tokenizer进行编码
            input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, text)
            input_ids = seq_padding(self.tokenizer, input_ids)
            token_type_ids = seq_padding(self.tokenizer, token_type_ids)
            # 标签形状为 (batch_size, 1)
            label = label.unsqueeze(1)
            # 需要 LongTensor
            input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
            # 梯度清零
            self.optimizer.zero_grad()
            # 迁移到GPU
            input_ids, token_type_ids, label = input_ids.to(self.device), token_type_ids.to(self.device), label.to(
                self.device)
            output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label) # 这里不需要labels
            # BertForSequenceClassification的输出loss和logits
            # BertModel原本的模型输出是last_hidden_state，pooler_output
            # bert_cnn的输出是[batch_size, num_class]

            y_pred_prob = output[1]
            y_pred_label = y_pred_prob.argmax(dim=1)
            # 计算loss
            # 这个 loss 和 output[0] 是一样的
            loss = self.criterion(y_pred_prob.view(-1, 2), label.view(-1))
            # loss = output[0]
            # 计算acc
            acc = ((y_pred_label == label.view(-1)).sum()).item()
            # 反向传播
            loss.backward()
            self.optimizer.step()
            # epoch 中的 loss 和 acc 累加
            epoch_loss += loss.item()
            epoch_acc += acc
            if i % 20 == 0:
                print("current loss:", epoch_loss / (i + 1), "\t", "current acc:", epoch_acc / ((i + 1) * len(label)))


        return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)

    def evaluate(self, iterator, test = False, cur_max_acc=0):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad():
            for _, batch in enumerate(iterator):
                label = batch["label"]
                text = batch["text"]
                input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, text)
                input_ids = seq_padding(self.tokenizer, input_ids)
                token_type_ids = seq_padding(self.tokenizer, token_type_ids)
                label = label.unsqueeze(1)
                input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
                input_ids, token_type_ids, label = input_ids.to(self.device), token_type_ids.to(self.device), label.to(self.device)
                output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
                # 更改了以下部分
                # y_pred_label = output[1].argmax(dim=1)
                y_pred_prob = output[1]
                y_pred_label = y_pred_prob.argmax(dim=1)
                loss = output[0]
                # loss = self.criterion(y_pred_prob.view(-1, 2), label.view(-1))
                acc = ((y_pred_label == label.view(-1)).sum()).item()
                epoch_loss += loss.item()
                epoch_acc += acc
        cur_loss = epoch_loss / len(iterator)
        cur_acc = epoch_acc / len(iterator.dataset.dataset)
        if test:
            if cur_acc > cur_max_acc:
                cur_max_acc = cur_acc
                self.save_model(args)
            return cur_loss,cur_acc,cur_max_acc
        else:
            return cur_loss, cur_acc


    def train(self, args):
        sentiment_train_loader, sentiment_valid_loader,sentiment_test_loader = self.get_data()
        cur_max_acc = 0

        for i in range(1,args.epoch+1):
            train_loss, train_acc = self.train_an_epoch(sentiment_train_loader)
            print("epoch:{}, train loss:{:.3f}, train acc:{:.3f}".format(i,train_loss,train_acc))

            valid_loss, valid_acc = self.evaluate(sentiment_valid_loader,test = False)
            print("epoch:{}, valid loss:{:.3f}, valid acc:{:.3f}".format(i,valid_loss,valid_acc))

            test_loss, test_acc ,cur_max_acc = self.evaluate(sentiment_test_loader,test = True,cur_max_acc = cur_max_acc)
            print("epoch:{}, test loss:{:.3f}, test acc:{:.3f}".format(i,test_loss,test_acc))

        #self.save_model(args)



    def save_model(self, args=None):
        model_save_path = self.config.get("result", "model_save_path") + args.my_model+"_model.bin"
        config_save_path = self.config.get("result", "config_save_path") + args.my_model+"_config.json"
        vocab_save_path = self.config.get("result", "vocab_save_path") + args.my_model+"_vocab.txt"

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), model_save_path)
        # model_to_save.config.to_json_file(config_save_path) # !!!'bert_lr' object has no attribute 'config'
        # self.tokenizer.save_vocabulary(vocab_save_path)
        print("model saved...")


    def predict(self, sentence):
        # self.model.setup()
        self.model_setup()
        self.model.eval()
        # 转token后padding
        input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, sentence)
        input_ids = seq_padding(self.tokenizer, [input_ids])
        token_type_ids = seq_padding(self.tokenizer, [token_type_ids])
        # 需要 LongTensor
        input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
        # 梯度清零
        self.optimizer.zero_grad()
        # 迁移到GPU
        input_ids, token_type_ids = input_ids.to(self.device), token_type_ids.to(self.device)
        output = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
        # y_pred_prob:各个类别的概率
        y_pred_prob = output[0]
        # 取概率最大的标签
        y_pred_label = y_pred_prob.argmax(dim=1)

        # 将torch.tensor转换回int形式
        return y_pred_label.item()




    def freezeSeed(self):
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",default="./prev_trained_model/chinese-bert-wwm",type=str)
    parser.add_argument("--my_model",default="bert_CNN",type=str,help="[huggingface_default, bert_lr_last4layer, bert_CNN, bert_lr]")
    parser.add_argument("--epoch",default="3",type=int)
    parser.add_argument("--result_dir",default="./result",type=str)
    parser.add_argument("--do_train", default= True,action="store_true",help="Whether to run training.")
    parser.add_argument("--do_predict",default= True,action="store_true",help="Whether to run predictions on the test set.")
    args = parser.parse_args()


    classifier = transformers_bert_binary_classification(args)
    if args.do_train:
        classifier.train(args)

    if args.do_predict:
        print(classifier.predict("交通不好"))  # 0
        print(classifier.predict("这个书不推荐看"))  # 0
        print(classifier.predict("交通挺好的"))  # 1
        print(classifier.predict("这个衣服很漂亮"))  # 1
        print(classifier.predict("酒店的服务人员服务周到，细致，给人一种宾至如归的感觉"))  # 1
        print(classifier.predict("东西不错，以后还会买"))  # 1
        print(classifier.predict("堵车了"))  # 0

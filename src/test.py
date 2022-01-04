# -*- coding: utf-8 -*-


import torch
import transformers
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification
from tools.config import Config
from transformers import AdamW
from tools.utils import convert_text_to_ids, seq_padding


class bert_classifier(object):
    def __init__(self):
        self.config = Config()
        self.device_setup()
        self.model_setup()

    def device_setup(self):
        """
        设备配置并加载BERT模型
        :return:
        """

        # 使用GPU，通过model.to(device)的方式使用
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_save_path = self.config.get("result", "model_save_path")
        config_save_path = self.config.get("result", "config_save_path")
        vocab_save_path = self.config.get("result", "vocab_save_path")

        self.model_config = BertConfig.from_json_file(config_save_path)
        self.model = BertForSequenceClassification(self.model_config)
        self.state_dict = torch.load(model_save_path)
        self.model.load_state_dict(self.state_dict)
        self.tokenizer = transformers.BertTokenizer(vocab_save_path)
        self.model.to(self.device)
        self.model.eval()


    def model_setup(self):
        weight_decay = self.config.get("training_rule", "weight_decay")
        learning_rate = self.config.get("training_rule", "learning_rate")

        # 定义优化器和损失函数
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def predict(self, sentence):
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
        y_pred_prob = output[0]
        y_pred_label = y_pred_prob.argmax(dim=1)
        print(y_pred_label)

if __name__ == '__main__':
    predictor = bert_classifier()
    predictor.predict('测试测试测试')
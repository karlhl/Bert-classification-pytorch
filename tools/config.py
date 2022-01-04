# -*- coding: utf-8 -*-

"""
Created on 2020-07-29 09:03
@Author  : Justin Jiang
@Email   : jw_jiang@pku.edu.com

配置模型、路径、与训练相关参数
"""

class Config(object):
    def __init__(self):
        self.config_dict = {
            "data_path": {
                "trainingSet_path": "./data/sentiment/sentiment.train.data",
                "valSet_path": "./data/sentiment/sentiment.valid.data",
                "testingSet_path": "./data/sentiment/sentiment.test.data"
            },

            "BERT_path": {
                "file_path": './prev_trained_model/chinese-bert-wwm/',
                "config_path": './prev_trained_model/chinese-bert-wwm/',
                "vocab_path": './prev_trained_model/chinese-bert-wwm/',
            },

            "training_rule": {
                "max_length": 300, # 输入序列长度，别超过512
                "hidden_dropout_prob": 0.3,
                "num_labels": 2, # 几分类个数
                "learning_rate": 1e-5,
                "weight_decay": 1e-2,
                "batch_size": 64
            },

            "result": {
                "model_save_path": './result/',
                "config_save_path": './result/',
                "vocab_save_path": './result/'
            }
        }

    def get(self, section, name):
        return self.config_dict[section][name]
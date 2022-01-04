# -*- coding: utf-8 -*-
"""
手动实现transformer.models.bert.BertForSequenceClassification()函数
根据论文[How to Fine-Tune BERT for Text Classification（2019）](https://www.aclweb.org/anthology/P18-1031.pdf)
在分类问题上，把最后四层进行concat然后maxpooling 输出的结果会比直接输出最后一层的要好
这里进行实现测试

"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer
import torch.nn.functional as F



class bert_lr_last4layer_Config(nn.Module):
    def __init__(self):
        self.bert_path = "./prev_trained_model/chinese-bert-wwm"
        self.config_path = "./prev_trained_model/chinese-bert-wwm/config.json"

        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.num_labels = 2
        self.dropout_bertout = 0.2
        self.mytrainedmodel = "./result/bert_lr_last4layer_model.bin"
        """
        current loss: 0.4363991916179657 	 current acc: 0.8125
        current loss: 0.1328232882924341 	 current acc: 0.9527363184079602
        current loss: 0.11797185830000853 	 current acc: 0.9585411471321695
        train loss:  0.11880445411248554 	 train acc: 0.9583704495516361
        valid loss:  0.1511497257672476 	 valid acc: 0.9431549028896258
        """

class bert_lr_last4layer(nn.Module):

    def __init__(self,config):
        super(bert_lr_last4layer, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path,config = config.config_path)
        self.dropout_bertout = nn.Dropout(config.dropout_bertout)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # outputs = outputs[2] # [1]是pooled的结果 # [3]是hidden_states 12层
        hidden_states = outputs.hidden_states
        nopooled_output = torch.cat((hidden_states[9],hidden_states[10],hidden_states[11],hidden_states[12]),1)
        batch_size = nopooled_output.shape[0] # 32
        # print(batch_size)
        # print(nopooled_output.shape) # torch.Size([32, 400, 768])
        kernel_hight = nopooled_output.shape[1]
        pooled_output = F.max_pool2d(nopooled_output,kernel_size = (kernel_hight,1))
        # print(pooled_output.shape) # torch.Size([32, 1, 768])

        flatten = pooled_output.view(batch_size,-1)
        # print(flatten.shape) # [32,768]

        flattened_output = self.dropout_bertout(flatten)

        logits = self.classifier(flattened_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss,logits


if __name__ == "__main__":
    model = bert_lr_last4layer(config=bert_lr_last4layer_Config())
    print(model)
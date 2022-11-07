from torch import nn
from transformers import BertModel


class BertNSPPretrained(nn.Module):

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertNSPPretrained, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(bert_pretrained_model_dir, config)
        else:
            self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.classifier_weak = nn.Linear(config.hidden_size, 2)

    def forward(self,
                input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size] 如果输入模型的只有一个序列，那么这个参数也不用传值
                position_ids=None,
                next_phrase_labels=None,
                next_phrase_weak_labels=None):  # [batch_size,]
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)

        # pooled_output: [batch_size, hidden_size]
        seq_relationship_score = self.classifier(out[1])
        seq_relationship_score_weak = self.classifier_weak(out[1])
        # seq_relationship_score: [batch_size, 2]
        if next_phrase_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(seq_relationship_score.view(-1, 2), next_phrase_labels.view(-1))
            loss_weak = loss_fct(seq_relationship_score_weak.view(-1, 2), next_phrase_weak_labels.view(-1))
            return loss + loss_weak, seq_relationship_score, seq_relationship_score_weak
        else:
            return seq_relationship_score, seq_relationship_score_weak

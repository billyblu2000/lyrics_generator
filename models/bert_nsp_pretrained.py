import torch
from torch import nn
from transformers import BertModel


def get_activation(activation_string):
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Unsupported activation: %s" % act)


class BertForLMTransformHead(nn.Module):
    """
    用于BertForMaskedLM中的一次变换。 因为在单独的MLM任务中
    和最后NSP与MLM的整体任务中均要用到，所以这里单独抽象为一个类便于复用
    ref: https://github.com/google-research/bert/blob/master/run_pretraining.py
        第248-262行
    """

    def __init__(self, config, bert_model_embedding_weights=None):
        """
        :param config:
        :param bert_model_embedding_weights:
        the output-weights are the same as the input embeddings, but there is
        an output-only bias for each token. 即TokenEmbedding层中的词表矩阵
        """
        super(BertForLMTransformHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        if bert_model_embedding_weights is not None:
            self.decoder.weight = nn.Parameter(bert_model_embedding_weights)
        # [hidden_size, vocab_size]
        self.decoder.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        """
        :param hidden_states: [src_len, batch_size, hidden_size] Bert最后一层的输出
        :return:
        """
        hidden_states = self.dense(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.transform_act_fn(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.LayerNorm(hidden_states)  # [src_len, batch_size, hidden_size]
        hidden_states = self.decoder(hidden_states)
        # hidden_states:  [src_len, batch_size, vocab_size]
        return hidden_states


class BertNSPPretrained(nn.Module):

    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertNSPPretrained, self).__init__()
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(bert_pretrained_model_dir, config)
        else:
            self.bert = BertModel(config)
        weights = None
        if 'use_embedding_weight' in config.__dict__ and config.use_embedding_weight:
            # weights = None
            # weights = self.bert.bert_embeddings.word_embeddings.embedding.weight
            weights = self.bert._modules['embeddings'].word_embeddings.weight
        self.mlm_prediction = BertForLMTransformHead(config, weights)
        self.nsp_prediction = nn.Linear(config.hidden_size, config.nsp_num_classes)
        self.config = config

    def forward(self,
                input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len] mask掉padding部分的内容
                token_type_ids=None,  # [src_len, batch_size] 如果输入模型的只有一个序列，那么这个参数也不用传值
                position_ids=None,
                mlm_labels=None,
                next_phrase_labels=None):  # [batch_size,]
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)

        sequence_output = out[0]  # 取Bert最后一层的输出
        # sequence_output: [src_len, batch_size, hidden_size]
        mlm_prediction_logits = self.mlm_prediction(sequence_output)
        # mlm_prediction_logits: [src_len, batch_size, vocab_size]
        nsp_pred_logits = self.nsp_prediction(out[1])
        # nsp_pred_logits： [batch_size, 2]
        if mlm_labels is not None and next_phrase_labels is not None:
            loss_fct_mlm = nn.CrossEntropyLoss(ignore_index=0)
            # MLM任务在构造数据集时pandding部分和MASK部分都是用的0来填充，所以ignore_index需要指定为0
            loss_fct_nsp = nn.CrossEntropyLoss()
            # 由于NSP中的分类标签中含有0，上面MLM中的损失指定了ignore_index=0，所以这里需要重新定义一个CrossEntropyLoss
            # 如果MLM任务在padding和MASK中用100之类的来代替，那么两者可以共用一个CrossEntropyLoss
            mlm_loss = loss_fct_mlm(mlm_prediction_logits.reshape(-1, self.config.vocab_size),
                                    mlm_labels.reshape(-1))
            nsp_loss = loss_fct_nsp(nsp_pred_logits.reshape(-1, self.config.nsp_num_classes),
                                    next_phrase_labels.reshape(-1))
            total_loss = mlm_loss + nsp_loss
            return total_loss, mlm_prediction_logits, nsp_pred_logits
        else:
            return mlm_prediction_logits, nsp_pred_logits

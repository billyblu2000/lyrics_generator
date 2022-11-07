import os
import time
from copy import deepcopy

import torch
from torch.optim import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup, BertTokenizer

from models.bert_nsp_pretrain_utils.bert_nsp_pretrain_config import ModelConfig
from models.bert_nsp_pretrain_utils.bert_nsp_pretrain_dataloader import LoadBertPretrainingDataset
from models.bert_nsp_pretrain_utils.bert_nsp_pretrain_utils import random_init
from models.bert_nsp_pretrained import BertNSPPretrained


def train(config):
    model = BertNSPPretrained(config,
                              config.pretrained_model_dir)
    last_epoch = -1
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        last_epoch = checkpoint['last_epoch']
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        print('Load state dict')
    model = model.to(config.device)
    random_init(config)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDataset(vocab_path=config.vocab_path,
                                             tokenizer=bert_tokenize,
                                             batch_size=config.batch_size,
                                             max_sen_len=config.max_sen_len,
                                             max_position_embeddings=config.max_position_embeddings,
                                             pad_index=config.pad_index,
                                             is_sample_shuffle=config.is_sample_shuffle,
                                             val_set_portion=config.validation_set_portion,
                                             test_set_portion=config.test_set_portion,
                                             random_state=config.random_state,
                                             data_name=config.data_name,
                                             masked_rate=config.masked_rate,
                                             masked_token_rate=config.masked_token_rate,
                                             masked_token_unchanged_rate=config.masked_token_unchanged_rate)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(file_path=config.dataset_dir)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
            "initial_lr": config.learning_rate

        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "initial_lr": config.learning_rate
        },
    ]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          int(len(train_iter) * 0),
                                                          int(config.epochs * len(train_iter)),
                                                          last_epoch=last_epoch)
    max_acc = 0
    state_dict = None
    print(f'Start Training: epoch={config.epochs}, batch_size={config.batch_size}, total_steps={len(train_iter)/config.batch_size*config.epochs}')
    for epoch in range(config.epochs):
        print(f'Epoch {epoch}')
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_segs, b_mask, b_nsp_label, b_nsp_label_weak) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_nsp_label = b_nsp_label.to(config.device)
            b_nsp_label_weak = b_nsp_label_weak.to(config.device)
            loss, nsp_logits, nsp_weak_logits = model(input_ids=b_token_ids,
                                                      attention_mask=b_mask,
                                                      token_type_ids=b_segs,
                                                      next_phrase_labels=b_nsp_label,
                                                      next_phrase_weak_labels=b_nsp_label_weak)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses += loss.item()
            nsp_acc, _, _, nsp_weak_acc, _, _ = accuracy(nsp_logits, nsp_weak_logits, b_nsp_label, b_nsp_label_weak,
                                                         data_loader.PAD_IDX)
            if idx % 20 == 0:
                config.writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                config.writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)
                config.writer.add_scalars(main_tag='Training/Accuracy',
                                          tag_scalar_dict={'NSP': nsp_acc,
                                                           'NSP_weak': nsp_weak_acc},
                                          global_step=scheduler.last_epoch)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        if (epoch + 1) % config.model_val_per_epoch == 0:
            nsp_acc, nsp_weak_acc = evaluate(config, val_iter, model, data_loader.PAD_IDX)
            print(f'Epoch {epoch}: train_loss={train_loss}, nsp_acc={nsp_acc}, nsp_weak_acc={nsp_weak_acc}')
            config.writer.add_scalars(main_tag='Testing/Accuracy',
                                      tag_scalar_dict={'NSP': nsp_acc,
                                                       'NSP_weak': nsp_weak_acc},
                                      global_step=scheduler.last_epoch)
            # mlm_acc, nsp_acc = evaluate(config, train_iter, model, data_loader.PAD_IDX)
            if nsp_acc > max_acc:
                max_acc = nsp_acc
                state_dict = deepcopy(model.state_dict())
            torch.save({'last_epoch': scheduler.last_epoch,
                        'model_state_dict': state_dict},
                       config.model_save_path)


def accuracy(nsp_logits, nsp_weak_logits, nsp_label, nsp_weak_label, PAD_IDX):
    nsp_correct = (nsp_logits.argmax(1) == nsp_label).float().sum()
    nsp_total = len(nsp_label)
    nsp_acc = float(nsp_correct) / nsp_total
    nsp_weak_correct = (nsp_logits.argmax(1) == nsp_label).float().sum()
    nsp_weak_total = len(nsp_label)
    nsp_weak_acc = float(nsp_correct) / nsp_total
    return [nsp_acc, nsp_correct, nsp_total, nsp_weak_acc, nsp_weak_correct, nsp_weak_total]


def evaluate(config, data_iter, model, PAD_IDX):
    model.eval()
    nsp_corrects, nsp_totals, nsp_weak_corrects, nsp_weak_totals = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (b_token_ids, b_segs, b_mask, b_nsp_label, b_nsp_label_weak) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_nsp_label = b_nsp_label.to(config.device)
            b_nsp_label_weak = b_nsp_label_weak.to(config.device)
            nsp_logits, nsp_weak_logits = model(input_ids=b_token_ids,
                                                attention_mask=b_mask,
                                                token_type_ids=b_segs)
            result = accuracy(nsp_logits, nsp_weak_logits, b_nsp_label, b_nsp_label_weak, PAD_IDX)
            _, nsp_cor, nsp_tot, _, nsp_weak_cor, nsp_weak_tot = result
            nsp_corrects += nsp_cor
            nsp_totals += nsp_tot
            nsp_weak_corrects += nsp_weak_cor
            nsp_weak_totals += nsp_weak_tot
    model.train()
    return [float(nsp_corrects) / nsp_totals, float(nsp_weak_corrects) / nsp_weak_totals]


def inference(config, sentences=None, masked=False, language='en', random_state=None):
    """
    :param config:
    :param sentences:
    :param masked: 推理时的句子是否Mask
    :param language: 语种
    :param random_state:  控制mask字符时的随机状态
    :return:
    """
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadBertPretrainingDataset(vocab_path=config.vocab_path,
                                             tokenizer=bert_tokenize,
                                             pad_index=config.pad_index,
                                             random_state=config.random_state,
                                             masked_rate=0.15)  # 15% Mask掉
    token_ids, pred_idx, mask = data_loader.make_inference_samples(sentences,
                                                                   masked=masked,
                                                                   language=language,
                                                                   random_state=random_state)
    model = BertNSPPretrained(config,
                              config.pretrained_model_dir)
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
    else:
        raise ValueError(f"模型 {config.model_save_path} 不存在！")
    model = model.to(config.device)
    model.eval()
    with torch.no_grad():
        token_ids = token_ids.to(config.device)  # [src_len, batch_size]
        mask = mask.to(config.device)
        mlm_logits, _ = model(input_ids=token_ids,
                              attention_mask=mask)
    pretty_print(token_ids, mlm_logits, pred_idx,
                 data_loader.vocab.itos, sentences, language)


def pretty_print(token_ids, logits, pred_idx, itos, sentences, language):
    """
    格式化输出结果
    :param token_ids:   [src_len, batch_size]
    :param logits:  [src_len, batch_size, vocab_size]
    :param pred_idx:   二维列表，每个内层列表记录了原始句子中被mask的位置
    :param itos:
    :param sentences: 原始句子
    :return:
    """
    token_ids = token_ids.transpose(0, 1)  # [batch_size,src_len]
    logits = logits.transpose(0, 1)  # [batch_size, src_len,vocab_size]
    y_pred = logits.argmax(axis=2)  # [batch_size, src_len]
    sep = " " if language == 'en' else ""
    for token_id, sentence, y, y_idx in zip(token_ids, sentences, y_pred, pred_idx):
        sen = [itos[id] for id in token_id]
        sen_mask = sep.join(sen).replace(" ##", "").replace("[PAD]", "").replace(" ,", ",")
        sen_mask = sen_mask.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()
        for idx in y_idx:
            sen[idx] = itos[y[idx]].replace("##", "")
        sen = sep.join(sen).replace("[PAD]", "").replace(" ,", ",")
        sen = sen.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()


if __name__ == '__main__':
    config = ModelConfig()
    train(config)

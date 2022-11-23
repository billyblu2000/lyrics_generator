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
        checkpoint = torch.load(config.model_save_path, map_location=config.device)
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
                                             masked_token_unchanged_rate=config.masked_token_unchanged_rate,
                                             nsp_num_classes=config.nsp_num_classes,
                                             only_mlm_task=config.only_mlm_task)
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
    print(
        f'Start Training: epoch={config.epochs}, batch_size={config.batch_size}, total_steps={len(train_iter) / config.batch_size * config.epochs}')
    for epoch in range(config.epochs):
        print(f'Epoch {epoch}')
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            b_nsp_label = b_nsp_label.to(config.device)
            loss, mlm_logits, nsp_logits = model(input_ids=b_token_ids,
                                                 attention_mask=b_mask,
                                                 token_type_ids=b_segs,
                                                 mlm_labels=b_mlm_label,
                                                 next_phrase_labels=b_nsp_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses += loss.item()
            mlm_acc, _, _, nsp_acc, _, _ = accuracy(mlm_logits, nsp_logits,
                                                    b_mlm_label, b_nsp_label,
                                                    data_loader.PAD_IDX)
            if idx % 20 == 0:
                config.writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                config.writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)
                config.writer.add_scalars(main_tag='Training/Accuracy',
                                          tag_scalar_dict={'MLM': mlm_acc,
                                                           'NSP': nsp_acc},
                                          global_step=scheduler.last_epoch)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        if (epoch + 1) % config.model_val_per_epoch == 0:
            mlm_acc, nsp_acc = do_evaluate(config, val_iter, model, data_loader.PAD_IDX)
            print(
                f'Epoch {epoch}: train_loss={train_loss}, mlm_acc={mlm_acc}, nsp_acc={nsp_acc}')
            config.writer.add_scalars(main_tag='Testing/Accuracy',
                                      tag_scalar_dict={'MLM': mlm_acc,
                                                       'NSP': nsp_acc},
                                      global_step=scheduler.last_epoch)
            # mlm_acc, nsp_acc = evaluate(config, train_iter, model, data_loader.PAD_IDX)
            if nsp_acc > max_acc:
                max_acc = nsp_acc
                state_dict = deepcopy(model.state_dict())
            torch.save({'last_epoch': scheduler.last_epoch,
                        'model_state_dict': state_dict},
                       config.model_save_path)


def evaluate(config):
    model = BertNSPPretrained(config,
                              config.pretrained_model_dir)
    last_epoch = -1
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path, map_location=config.device)
        last_epoch = checkpoint['last_epoch']
        loaded_paras = checkpoint['model_state_dict']
        model.load_state_dict(loaded_paras)
        print('Load state dict')
    model = model.to(config.device)
    random_init(config)
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
                                             masked_token_unchanged_rate=config.masked_token_unchanged_rate,
                                             nsp_num_classes=config.nsp_num_classes,
                                             only_mlm_task=config.only_mlm_task)
    train_iter, test_iter, val_iter = data_loader.load_train_val_test_data(file_path=config.dataset_dir)
    mlm_acc, nsp_acc = do_evaluate(config, val_iter, model, data_loader.PAD_IDX)
    return mlm_acc, nsp_acc


def accuracy(mlm_logits, nsp_logits, mlm_labels, nsp_label, PAD_IDX):
    mlm_pred = mlm_logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [src_len,batch_size,src_vocab_size] 转成 [batch_size, src_len,src_vocab_size]
    mlm_true = mlm_labels.transpose(0, 1).reshape(-1)
    # 将 [src_len,batch_size] 转成 [batch_size， src_len]
    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况，得到预测正确的个数（此时还包括有mask位置）
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉mlm_acc中mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total
    nsp_correct = (nsp_logits.argmax(1) == nsp_label).float().sum()
    nsp_total = len(nsp_label)
    nsp_acc = float(nsp_correct) / nsp_total
    return [mlm_acc, mlm_correct, mlm_total,
            nsp_acc, nsp_correct, nsp_total]


def do_evaluate(config, data_iter, model, PAD_IDX):
    model.eval()
    mlm_corrects, mlm_totals, nsp_corrects, nsp_totals = 0, 0, 0, 0
    with torch.no_grad():
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_nsp_label) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            b_nsp_label = b_nsp_label.to(config.device)
            mlm_logits, nsp_logits = model(input_ids=b_token_ids,
                                           attention_mask=b_mask,
                                           token_type_ids=b_segs)
            result = accuracy(mlm_logits, nsp_logits,
                              b_mlm_label, b_nsp_label, PAD_IDX)
            _, mlm_cor, mlm_tot, _, nsp_cor, nsp_tot = result
            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot
            nsp_corrects += nsp_cor
            nsp_totals += nsp_tot
            print(float(nsp_corrects) / nsp_totals)
    model.train()
    return [float(mlm_corrects) / mlm_totals,
            float(nsp_corrects) / nsp_totals]


if __name__ == '__main__':
    config = ModelConfig()
    # train(config)
    evaluate(config)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import random
from transformers import BertTokenizer


class JointDataset(torch.utils.data.Dataset):
    def __init__(self, mlm_dataset, kg_dataset, account_triples_idx):
        self.mlm_dataset = mlm_dataset
        self.kg_dataset = kg_dataset
        self.account_triples_idx = account_triples_idx

    def __len__(self):
        return len(self.mlm_dataset)

    def __getitem__(self, idx):
        mlm_item = self.mlm_dataset[idx]
        kg_idx = self.account_triples_idx[idx][0]
        triple_pos, triple_neg = self.kg_dataset[kg_idx]

        return {
            "input_ids": mlm_item["input_ids"],
            "attention_mask": mlm_item["attention_mask"],
            "labels": mlm_item["labels"],
            "triple_pos": triple_pos,
            "triple_neg": triple_neg
        }


class Bm25BertDataset(torch.utils.data.Dataset):
    def __init__(self, texts, doc_token_bm25_scores, tokenizer, bm25_top_p=0.3, max_length=64):
        self.texts = texts
        self.scores = doc_token_bm25_scores
        self.tokenizer = tokenizer
        self.bm25_top_p = bm25_top_p
        self.max_length = max_length
        all_scores = np.concatenate(self.scores)
        self.bm25_threshold = np.percentile(all_scores, 100 * (1.0 - bm25_top_p))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        token_scores = self.scores[idx]
        tokens = text
        masked_tokens = []
        labels = []

        for token_idx, token in enumerate(tokens):
            score = token_scores[token_idx]
            prob_mask = 0.1
            if score >= self.bm25_threshold:
                prob_mask = 1.0

            do_mask = (random.random() < prob_mask)

            if do_mask:
                rand_val = random.random()
                if rand_val < 0.8:
                    masked_tokens.append("[MASK]")
                elif rand_val < 0.9:
                    masked_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                else:
                    masked_tokens.append(token)
                labels.append(token)
            else:
                masked_tokens.append(token)
                labels.append("[PAD]")

        encoding = self.tokenizer(
            masked_tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        with self.tokenizer.as_target_tokenizer():
            label_encoding = self.tokenizer(
                labels,
                is_split_into_words=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        mlm_labels = label_encoding["input_ids"].squeeze(0)
        pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        mlm_labels = torch.where(mlm_labels == pad_id, torch.tensor(-100), mlm_labels)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": mlm_labels
        }


class KGLP_Dataset(torch.utils.data.Dataset):
    def __init__(self, triples, num_entities):
        self.triples = triples
        self.num_entities = num_entities

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        else:
            return self._get_single_item(idx)

    def _get_single_item(self, idx):
        h, r, t = self.triples[idx]
        if random.random() < 0.5:
            t_neg = random.randint(0, self.num_entities - 1)
            while t_neg == t:
                t_neg = random.randint(0, self.num_entities - 1)
            return (h, r, t), (h, r, t_neg)
        else:
            h_neg = random.randint(0, self.num_entities - 1)
            while h_neg == h:
                h_neg = random.randint(0, self.num_entities - 1)
            return (h, r, t), (h_neg, r, t)

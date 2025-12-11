import torch.nn as nn
import torch
import os
from transformers import BertTokenizer
from transformers import BertForMaskedLM


class KGETHBERT(nn.Module):
    def __init__(self,
                 bert_model_name="bert-base-uncased",
                 num_entities=10,
                 num_relations=5,
                 emb_dim=128):
        super().__init__()

        self.bert_mlm = BertForMaskedLM.from_pretrained(bert_model_name)

        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)

        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.margin = 0.1

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                triple_pos=None,
                triple_neg=None):

        mlm_out = self.bert_mlm(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                return_dict=True)
        mlm_loss = mlm_out.loss

        if triple_pos is not None and triple_neg is not None:
            h_pos, r_pos, t_pos = triple_pos[:, 0], triple_pos[:, 1], triple_pos[:, 2]
            h_neg, r_neg, t_neg = triple_neg[:, 0], triple_neg[:, 1], triple_neg[:, 2]

            h_pos_e = self.entity_emb(h_pos)
            r_pos_e = self.relation_emb(r_pos)
            t_pos_e = self.entity_emb(t_pos)

            h_neg_e = self.entity_emb(h_neg)
            r_neg_e = self.relation_emb(r_neg)
            t_neg_e = self.entity_emb(t_neg)

            pos_score = torch.norm(h_pos_e + r_pos_e - t_pos_e, p=2, dim=1)
            neg_score = torch.norm(h_neg_e + r_neg_e - t_neg_e, p=2, dim=1)

            zeros = torch.zeros_like(pos_score)
            kg_loss = torch.mean(torch.maximum(self.margin + pos_score - neg_score, zeros))
        else:
            kg_loss = 0.0

        total_loss = mlm_loss + kg_loss

        return total_loss, mlm_loss, kg_loss

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.bert_mlm.save_pretrained(save_directory)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))

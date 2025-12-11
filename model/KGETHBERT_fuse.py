import torch
import torch.nn as nn
import os
from transformers import BertTokenizer

from model.BertForMaskedLM import BertForMaskedLM
from model.Attention import CLSNodeAttentionLayer


class KGETHBERT(nn.Module):
    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        cls_dim=768,
        node_dim=768,
        num_relations=5,
        margin=0.1,
        num_heads=2
    ):
        super().__init__()

        self.bert_mlm = BertForMaskedLM.from_pretrained(bert_model_name)

        self.cls_node_attn = CLSNodeAttentionLayer(
            cls_dim=cls_dim,
            node_dim=node_dim,
            num_heads=num_heads
        )

        self.num_relations = num_relations
        self.rel_emb_dim = node_dim
        self.relation_emb = nn.Embedding(num_relations, self.rel_emb_dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)

        self.margin = margin

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        node_repr=None,
        triple_pos=None,
        triple_neg=None,
    ):
        outputs = self.bert_mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            return_dict=True
        )
        last_hidden = outputs.hidden_states

        cls_hidden = last_hidden[:, 0:1, :]
        fused_cls, fused_nodes = self.cls_node_attn(cls_hidden, node_repr)

        hidden_states_new = last_hidden.clone()
        hidden_states_new[:, 0:1, :] = fused_cls
        mlm_loss, _ = self.bert_mlm.mlm_from_hidden(
            hidden_states=hidden_states_new,
            labels=labels
        )

        if mlm_loss is None:
            mlm_loss = torch.tensor(0.0, device=last_hidden.device)

        link_loss = torch.tensor(0.0, device=last_hidden.device)
        if triple_pos is not None and triple_neg is not None:
            h_pos_e = self._gather_node_embedding(fused_nodes, triple_pos[:,0])
            t_pos_e = self._gather_node_embedding(fused_nodes, triple_pos[:,2])
            h_neg_e = self._gather_node_embedding(fused_nodes, triple_neg[:,0])
            t_neg_e = self._gather_node_embedding(fused_nodes, triple_neg[:,2])

            r_pos = triple_pos[:,1]
            r_neg = triple_neg[:,1]
            r_pos_e = self.relation_emb(r_pos)
            r_neg_e = self.relation_emb(r_neg)

            pos_score = torch.norm(h_pos_e + r_pos_e - t_pos_e, p=2, dim=1)
            neg_score = torch.norm(h_neg_e + r_neg_e - t_neg_e, p=2, dim=1)
            zeros = torch.zeros_like(pos_score)
            link_loss = torch.mean(torch.maximum(self.margin + pos_score - neg_score, zeros))

        total_loss = mlm_loss + link_loss
        return total_loss, mlm_loss, link_loss, fused_cls, fused_nodes

    def _gather_node_embedding(self, fused_nodes, index):
        bs, max_nodes, node_dim = fused_nodes.shape
        batch_indices = torch.arange(bs, device=fused_nodes.device)
        out = fused_nodes[batch_indices, index, :]
        return out

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.bert_mlm.save_pretrained(save_directory)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.save_pretrained(save_directory)

        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))

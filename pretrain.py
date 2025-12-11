import torch
import datetime
import os
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from model.KGETHBERT_Model import KGETHBERT
from model.Dataset import JointDataset, Bm25BertDataset, KGLP_Dataset
from utils import load_pkl
import csv
import atexit


def _resolve_b4e_base():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(root, 'Dataset', 'B4E'),
        os.path.join(root, 'KGBERT4ETH', 'Dataset', 'B4E'),
        os.path.join(root, 'Dataset', 'B4E'),
    ]
    for base in candidates:
        if os.path.exists(base):
            return base
    return os.path.join(root, 'Dataset', 'B4E')


account_texts = load_pkl('dev_corpus.pkl')
try:
    doc_token_bm25_scores = load_pkl('list_token_bm25_scores.pkl')
except Exception:
    base = _resolve_b4e_base()
    preprocess_dir = os.path.join(base, 'preprocess')
    train_tsv = os.path.join(preprocess_dir, 'train.tsv') if os.path.exists(os.path.join(preprocess_dir, 'train.tsv')) else os.path.join(base, 'train.tsv')
    val_tsv = os.path.join(preprocess_dir, 'validation.tsv') if os.path.exists(os.path.join(preprocess_dir, 'validation.tsv')) else os.path.join(base, 'validation.tsv')

    def read_indices(tsv_path):
        csv.field_size_limit(10000000)
        idxs = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            r = csv.reader(f, delimiter='\t')
            next(r)
            for row in r:
                idxs.append(int(row[0]))
        return idxs

    train_idxs = read_indices(train_tsv)
    val_idxs = read_indices(val_tsv)
    import pickle
    dt_path = os.path.join(base, 'train_list_token_bm25_scores.pkl')
    dv_path = os.path.join(base, 'validation_list_token_bm25_scores.pkl')
    dt = pickle.load(open(dt_path, 'rb'))
    dv = pickle.load(open(dv_path, 'rb'))
    merged = dict(dt)
    for k, v in dv.items():
        if k not in merged:
            merged[k] = v
    doc_token_bm25_scores = [merged[i] for i in (train_idxs + val_idxs)]

kg_triples = load_pkl('triples.pkl')
account_triples_idx = load_pkl('account_triples_indices.pkl')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

relation_set = set()
entity_set = set()

for (h, r, t) in kg_triples:
    relation_set.add(r)
    entity_set.add(h)
    entity_set.add(t)

relation2id = {rel: idx for idx, rel in enumerate(sorted(relation_set))}

num_entities = len(entity_set)
num_relations = len(relation2id)

kg_triples_id = [(h, relation2id[r], t) for (h, r, t) in kg_triples]

n = min(len(account_texts), len(doc_token_bm25_scores), len(account_triples_idx))
account_texts = account_texts[:n]
doc_token_bm25_scores = doc_token_bm25_scores[:n]
account_triples_idx = account_triples_idx[:n]
_adj_scores = []
for i in range(n):
    s = doc_token_bm25_scores[i]
    l = len(account_texts[i])
    _adj_scores.append(s[:l] if len(s) >= l else (s + [0.0] * (l - len(s))))
doc_token_bm25_scores = _adj_scores

mlm_dataset = Bm25BertDataset(
    texts=account_texts,
    doc_token_bm25_scores=doc_token_bm25_scores,
    tokenizer=tokenizer,
    bm25_top_p=0.3,
    max_length=64
)

kg_dataset = KGLP_Dataset(triples=kg_triples_id, num_entities=num_entities)

joint_dataset = JointDataset(mlm_dataset, kg_dataset, account_triples_idx)
joint_dataloader = DataLoader(joint_dataset, batch_size=256, shuffle=False)

model = KGETHBERT(
    bert_model_name="bert-base-uncased",
    num_entities=num_entities,
    num_relations=num_relations,
    emb_dim=32
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

current_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
save_dir = f"Train_output/{current_time}"
os.makedirs(save_dir, exist_ok=True)
print(f"save_dir={save_dir}")

def _save():
    try:
        model.save_pretrained(save_dir)
        print(f"saved to {save_dir}")
    except Exception as e:
        print(str(e))

atexit.register(_save)

EPOCHS = 2
try:
    for epoch in range(EPOCHS):
        model.train()
        total_loss_val = 0.0
        total_mlm_loss_val = 0.0
        total_kg_loss_val = 0.0

        for batch in tqdm(joint_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            triple_pos = batch["triple_pos"] if isinstance(batch["triple_pos"], torch.Tensor) else torch.tensor(batch["triple_pos"], dtype=torch.long)
            triple_neg = batch["triple_neg"] if isinstance(batch["triple_neg"], torch.Tensor) else torch.tensor(batch["triple_neg"], dtype=torch.long)
            triple_pos = triple_pos.to(device)
            triple_neg = triple_neg.to(device)

            optimizer.zero_grad()
            total_loss, mlm_loss, kg_loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                triple_pos=triple_pos,
                triple_neg=triple_neg
            )

            total_loss.backward()
            optimizer.step()

            total_loss_val += total_loss.item()
            total_mlm_loss_val += mlm_loss.item()
            total_kg_loss_val += kg_loss.item()

        avg_loss = total_loss_val / len(joint_dataloader)
        avg_mlm_loss = total_mlm_loss_val / len(joint_dataloader)
        avg_kg_loss = total_kg_loss_val / len(joint_dataloader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, total_loss={avg_loss:.4f}, mlm_loss={avg_mlm_loss:.4f}, kg_loss={avg_kg_loss:.4f}"
        )

        _save()
except Exception as e:
    print(str(e))
    _save()

_save()

print("end")

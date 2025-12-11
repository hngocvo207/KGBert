import pickle as pkl
import tqdm

with open("deanrole_tmp_data/fromdeanrole1.pkl", "rb") as f:
    trans_seq = pkl.load(f)

next_trans_seq = {}
for address, transactions in tqdm.tqdm(trans_seq.items(), desc="处理交易数据"):

    trans_seq_str = ""
    for transaction in transactions:
        trans_str = f" amount: {transaction[0]} in_out: {transaction[1]} 2-gram: {transaction[2]} 3-gram: {transaction[3]} 4-gram: {transaction[4]} 5-gram: {transaction[5] } "
        trans_seq_str += trans_str

    trans_seq_str = trans_seq_str.strip() + '.'
    next_trans_seq[address] = [trans_seq_str]


with open("deanrole_tmp_data/fromdeanrole2.pkl", "wb") as f:
    pkl.dump(next_trans_seq, f)

print(1)
import pickle as pkl
from tqdm import tqdm 

with open(r'D:/SOTA/TLM/B4E/gen_b4e_seq/process_data/data_train_b4e/bert4eth_trans1.pkl', "rb") as f:
    trans_seq = pkl.load(f)

for address, transactions in tqdm(trans_seq.items(), desc="Sorting transaction lists"):
    # Sort transactions by timestamp (the second element of each transaction)
    sorted_transactions = sorted(transactions, key=lambda x: x[1])
    # Update the transactions for the current address with the sorted list
    trans_seq[address] = sorted_transactions

with open(r'D:/SOTA/TLM/B4E/gen_b4e_seq/process_data/data_train_b4e/bert4eth_trans2.pkl', "wb") as f:
    pkl.dump(trans_seq, f)

print(1)

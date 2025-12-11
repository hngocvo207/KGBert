import pickle as pkl
from tqdm import tqdm

with open(r'D:/SOTA/TLM/B4E/gen_b4e_seq/process_data/data_train_b4e/bert4eth_trans3.pkl', "rb") as f:
    trans_seq = pkl.load(f)


with open(r'D:/SOTA/TLM/Raw_Data/B4E/phisher_account.txt', "r") as f:
    addresses_in_file = {line.strip() for line in f}

next_trans_seq = {}

for address, transactions in tqdm(trans_seq.items() , desc='Calculating n-grams'):
    new_transactions = []
    for i in range(len(transactions)):
        n_gram = []
        for j in range(1,5):
            if i <= j: 
                n_gram.append(0)
            else:
                n_gram.append(transactions[i][1] - transactions[i-j][1])

        transaction = [[transactions[i][2], transactions[i][3]] + n_gram]
        new_transactions += transaction
    trans_seq[address] = new_transactions                



for address, transactions in tqdm(trans_seq.items(), desc='Generating transaction sequences'):
    prefix = '1' if address in addresses_in_file else '0'

    trans_seq_str = prefix
    for transaction in transactions:
        trans_str = f" amount: {transaction[0]} in_out: {transaction[1]} 2-gram: {transaction[2]} 3-gram: {transaction[3]} 4-gram: {transaction[4]} 5-gram: {transaction[5] } "
        trans_seq_str += trans_str
    trans_seq_str = trans_seq_str.strip() + '.'
    next_trans_seq[address] = [trans_seq_str]

with open(r'D:/SOTA/TLM/B4E/gen_b4e_seq/process_data/data_train_b4e/bert4eth_trans4.pkl', "wb") as f:
    pkl.dump(next_trans_seq, f)

print(1)

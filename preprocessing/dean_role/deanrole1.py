import pickle as pkl
import tqdm

with open("dean_role_trans_seq.pkl", "rb") as f:
    trans_seq = pkl.load(f)

for address, transactions in trans_seq.items():
    # Sort transactions by timestamp (the second element of each transaction)
    sorted_transactions = sorted(transactions, key=lambda x: x[1])
    # Update the transactions for the current address with the sorted list
    trans_seq[address] = sorted_transactions

for address, transactions in tqdm.tqdm(trans_seq.items()):
    for i in range(len(transactions)):
        # Initialize a list to hold the n-gram time differences for this transaction
        n_gram_times = []
        for n in range(2, 6):
            if i >= n - 1:

                n_gram_time_diff = transactions[i][1]-transactions[i - n + 1][1]
                n_gram_times.append(n_gram_time_diff)
            else:

                n_gram_times.append(0)

        transactions[i].extend(n_gram_times)

for address, transactions in tqdm.tqdm(trans_seq.items()):
    for i in range(len(transactions)):

        transactions[i] = transactions[i][2:]

        if transactions[i][1] == 'IN':
            transactions[i][1] = 0
        elif transactions[i][1] == 'OUT':
            transactions[i][1] = 1

with open("deanrole_tmp_data/fromdeanrole1.pkl", "wb") as f:
    pkl.dump(trans_seq, f)

print(1)

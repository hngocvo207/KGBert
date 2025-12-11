import pickle
import tqdm

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def remove_tag_except_first(accounts):
    for address, transactions in accounts.items():
        for i in range(1, len(transactions)):
            if 'tag' in transactions[i]:
                del transactions[i]['tag']

def merge_transactions(accounts):
    for address in accounts.keys():
        if accounts[address]:
            first_tag = accounts[address][0]['tag']
            merged_data = {'tag': first_tag, 'transactions': accounts[address]}
            accounts[address] = [merged_data]

input_path5 = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions6.pkl'
output_path5 = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions7.pkl'
accounts_data = load_data(input_path5)

remove_tag_except_first(accounts_data)

merge_transactions(accounts_data)

save_data(accounts_data, output_path5)



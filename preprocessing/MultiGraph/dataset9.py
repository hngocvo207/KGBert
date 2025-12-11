import pickle
import tqdm

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def remove_tag_from_transactions(accounts):
    for address, transactions in accounts.items():
        for transaction in transactions:
            for sub_transaction in transaction['transactions']:
                if 'tag' in sub_transaction:
                    del sub_transaction['tag']
input_path6 = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions8.pkl'
output_path6 = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions9.pkl'
accounts_data = load_data(input_path6)

remove_tag_from_transactions(accounts_data)

save_data(accounts_data, output_path6)


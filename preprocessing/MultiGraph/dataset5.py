import pickle
import tqdm

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def remove_fields(accounts, fields):
    for address in tqdm.tqdm(accounts.keys(), desc="删除字段"):
        for transaction in accounts[address]:
            for field in fields:
                if field in transaction:
                    del transaction[field]

input_path3 = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions4.pkl'
output_path3 = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions5.pkl'

accounts_data = load_data(input_path3)

fields_to_remove = ['from_address', 'to_address', 'timestamp']

remove_fields(accounts_data, fields_to_remove)

save_data(accounts_data, output_path3)


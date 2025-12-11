import pickle
import tqdm

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def convert_transactions_to_text(accounts):
    for address, transactions in accounts.items():
        for idx, transaction in enumerate(transactions):
            tag = transaction['tag']
            transaction_descriptions = []
            for sub_transaction in transaction['transactions']:
                description = ' '.join([f"{key}: {sub_transaction[key]}" for key in sub_transaction])
                transaction_descriptions.append(description)
            transactions[idx] = f"{tag} {'  '.join(transaction_descriptions)}."

input_path7 = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions9.pkl'
output_path7 = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions10.pkl'
accounts_data = load_data(input_path7)

convert_transactions_to_text(accounts_data)

save_data(accounts_data, output_path7)


import pickle


def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def sort_transactions_by_timestamp(accounts):
    sorted_accounts = {}
    for address, transactions in accounts.items():
        sorted_accounts[address] = sorted(transactions, key=lambda x: x['timestamp'])
    return sorted_accounts

input_path = r"D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions2.pkl"
output_path = r"D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions3.pkl"
accounts_data = load_data(input_path)

sorted_accounts_data = sort_transactions_by_timestamp(accounts_data)

save_data(sorted_accounts_data, output_path)

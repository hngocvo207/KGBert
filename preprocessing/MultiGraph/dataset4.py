import pickle
import tqdm

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def add_n_grams(accounts):
    for address, transactions in tqdm.tqdm(accounts.items()):
        for n in range(2, 6):
            gram_key = f"{n}-gram"
            for i in range(len(transactions)):
                if i < n-1:
                    transactions[i][gram_key] = 0
                else:
                    transactions[i][gram_key] = transactions[i]['timestamp'] - transactions[i-n+1]['timestamp']

input_path2 = r"D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions3.pkl"
output_path2 = r"D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions4.pkl"
accounts_data = load_data(input_path2)

add_n_grams(accounts_data)

save_data(accounts_data, output_path2)


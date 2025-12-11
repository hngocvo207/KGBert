import pickle
import random
import tqdm

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def shuffle_transactions(accounts):
    for address in tqdm.tqdm(accounts.keys()):
        random.shuffle(accounts[address])

input_path4 = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions5.pkl'
output_path4 = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions6.pkl'
# 加载数据
accounts_data = load_data(input_path4)

# 打乱交易数据
shuffle_transactions(accounts_data)

# 保存数据
save_data(accounts_data, output_path4)


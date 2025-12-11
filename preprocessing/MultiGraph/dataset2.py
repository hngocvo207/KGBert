import pickle
import os

def load_data(filename):
    # Kiểm tra xem file có tồn tại không trước khi mở
    if not os.path.exists(filename):
        print(f"LỖI: Không tìm thấy file tại đường dẫn: {filename}")
        print("-> Hãy kiểm tra lại xem file transactions1.pkl đang nằm ở đâu!")
        return None
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def process_transactions(transactions):
    if transactions is None: return # Dừng nếu không load được dữ liệu

    accounts = {}

    for tx in transactions:

        from_address = tx['from_address']
        if from_address not in accounts:
            accounts[from_address] = []
        accounts[from_address].append({**tx, 'in_out': 1})

        to_address = tx['to_address']
        if to_address not in accounts:
            accounts[to_address] = []
        accounts[to_address].append({**tx, 'in_out': 0})

    return accounts

#transactions = load_data('D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions1.pkl')

#processed_data = process_transactions(transactions)

#save_data(processed_data, 'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions2.pkl')

input_filename = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions1.pkl' 
output_filename = r'D:/Ngọc/FDA-NEU/Lab CV/SOTA data/KGBERT4ETH/NewDataset/transactions2.pkl'

# Gọi hàm5
print(f"Đang cố gắng đọc file: {input_filename} ...")
transactions = load_data(input_filename)

if transactions is not None:
    processed_data = process_transactions(transactions)
    save_data(processed_data, output_filename)
    print("HOÀN TẤT!")
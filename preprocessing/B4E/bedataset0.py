import numpy as np
import pickle as pkl
import argparse
import functools
import os
import pandas as pd
from random import sample
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Data Processing with PyTorch")
parser.add_argument("--phisher", type=bool, default=True, help="whether to include phisher detection dataset.")
parser.add_argument("--dataset", type=str, default=None, help="which dataset to use")
parser.add_argument("--data_dir", type=str, default="./raw_data", help="directory to save the data")
args = parser.parse_args()


HEADER = 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'.split(",")

def cmp_udf(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])
    if time1 < time2:
        return -1
    elif time1 > time2:
        return 1
    else:
        return 0

def cmp_udf_reverse(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])

    if time1 < time2:
        return 1
    elif time1 > time2:
        return -1
    else:
        return 0

def load_data(f_in, f_out):
    eoa2seq_out = {}
    error_trans = []
    while True:
        trans = f_out.readline()
        if trans == "":
            break
        record = trans.split(",")
        trans_hash = record[0]
        block_number = int(record[3])
        from_address = record[5]
        to_address = record[6]
        value = int(record[7]) / (pow(10, 12))
        gas = int(record[8])
        gas_price = int(record[9])
        block_timestamp = int(record[11])
        if from_address == "" or to_address == "":
            error_trans.append(trans)
            continue
        try:
            eoa2seq_out[from_address].append([to_address, block_number, block_timestamp, value, "OUT", 1])
        except:
            eoa2seq_out[from_address] = [[to_address, block_number, block_timestamp, value, "OUT", 1]]

    eoa2seq_in = {}
    while True:
        trans = f_in.readline()
        if trans == "":
            break
        record = trans.split(",")
        block_number = int(record[3])
        from_address = record[5]
        to_address = record[6]
        value = int(record[7]) / (pow(10, 12))
        gas = int(record[8])
        gas_price = int(record[9])
        block_timestamp = int(record[11])
        if from_address == "" or to_address == "":
            error_trans.append(trans)
            continue
        try:
            eoa2seq_in[to_address].append([from_address, block_number, block_timestamp, value, "IN", 1]) # not process trans
        except:
            eoa2seq_in[to_address] = [[from_address, block_number, block_timestamp, value, "IN", 1]] # in/out, cnt
    return eoa2seq_in, eoa2seq_out

def seq_generation(eoa2seq_in, eoa2seq_out):

    eoa_list = list(eoa2seq_out.keys()) # eoa_list must include eoa account only (i.e., have out transaction at least)
    eoa2seq = {}
    for eoa in eoa_list:
        out_seq = eoa2seq_out[eoa]
        try:
            in_seq = eoa2seq_in[eoa]
        except:
            in_seq = []
        seq_agg = sorted(out_seq + in_seq, key=functools.cmp_to_key(cmp_udf_reverse))
        cnt_all = 0
        for trans in seq_agg:
            cnt_all += 1
            # if cnt_all >= 5 and cnt_all<=10000:
            if cnt_all > 2 and cnt_all<=10000:
                eoa2seq[eoa] = seq_agg
                break

    return eoa2seq


def main():

    f_in = open(os.path.join(args.data_dir,r"D:/SOTA/TLM/Raw_Data/B4E/normal_eoa_transaction_in_slice_1000K.csv"), "r")
    f_out = open(os.path.join(args.data_dir,r"D:/SOTA/TLM/Raw_Data/B4E/normal_eoa_transaction_out_slice_1000K.csv"), "r")
    print("Add normal account transactions.")
    eoa2seq_in, eoa2seq_out = load_data(f_in, f_out)

    eoa2seq_agg = seq_generation(eoa2seq_in, eoa2seq_out)

    if args.phisher:
        print("Add phishing..")
        phisher_f_in = open(os.path.join(args.data_dir,r"D:/SOTA/TLM/Raw_Data/B4E/phisher_transaction_in.csv"), "r")
        phisher_f_out = open(os.path.join(args.data_dir,r"D:/SOTA/TLM/Raw_Data/B4E/phisher_transaction_out.csv"), "r")
        phisher_eoa2seq_in, phisher_eoa2seq_out = load_data(phisher_f_in, phisher_f_out)
  
        phisher_eoa2seq_agg = seq_generation(phisher_eoa2seq_in, phisher_eoa2seq_out)
        eoa2seq_agg.update(phisher_eoa2seq_agg)

    print("statistics:")
    length_list = []
    for eoa in eoa2seq_agg.keys():
        seq = eoa2seq_agg[eoa]
        length_list.append(len(seq))

    length_list = np.array(length_list)
    print("Median:", np.median(length_list))
    print("Mean:", np.mean(length_list))
    print("Seq #:", len(length_list))

    "Sampling with ratio 5:5 for finetuning"
    df = pd.read_csv(os.path.join(args.data_dir,r"D:/SOTA/TLM/Raw_Data/B4E/phisher_account.txt"),names = ["accounts"])
    phisher_account = df.accounts.values
    normal_accounts = []
    abnormal_accounts = []
    for eoa in tqdm(eoa2seq_agg.keys(),desc="Sampling with ratio 5:5 for finetuning"):
        if eoa in phisher_account:
            abnormal_accounts.append(eoa)
        else:
            normal_accounts.append(eoa)
    
    random.seed(42)
    selected_normal_accounts = sample(normal_accounts, k = len(abnormal_accounts))

    final_selected_accounts = selected_normal_accounts + abnormal_accounts

    eoa2seq_final = {eoa:eoa2seq_agg[eoa] for eoa in final_selected_accounts}
    with open(r"D:/SOTA/TLM/B4E/gen_b4e_seq/process_data/data_train_b4e/eoa2seq.pkl", "wb") as f:
        pkl.dump(eoa2seq_final, f)


if __name__ == '__main__':
    main()
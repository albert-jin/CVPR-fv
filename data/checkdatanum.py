from configs import LabelUnion, VLabelDiv_list
import json
import os
from tqdm import tqdm


def label_check(data):
    label_numcounts = {vlabel:0 for vlabel in VLabelDiv_list}
    for d in tqdm(data):
        if d['label'] in LabelUnion and LabelUnion[d['label']] in VLabelDiv_list:
            label_numcounts[LabelUnion[d['label']]] += 1
        elif d['label'] in VLabelDiv_list:
            label_numcounts[d['label']] += 1
        else:
            raise Exception(f'error: there exists disappeared label: {d["label"]}.')
    print('\n'.join([f'{vlabel}: {label_numcounts[vlabel]}' for vlabel in label_numcounts])+'\n**********\n')


def read_jsonl(file):
    print('read from', file)
    inp = open(file, encoding='utf-8', mode='rt', )
    data = [json.loads(line.strip()) for line in inp.readlines()]
    # print('data @ first row:', data[0], '\ntype:', type(data), ' length:', len(data))
    print('length: ', len(data))
    inp.close()
    return data


if __name__ == '__main__':
    dir_path = '../data'
    for filename in os.listdir(dir_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(dir_path, filename)
            label_check(read_jsonl(file_path))


# D:\anaconda3\envs\protoco_win\python.exe "C:/Users/Weiqiang Jin/Desktop/fake detection/Det2Ver/Det2Ver/data/checkdatanum.py"
# load LLM Tokenizer...
# loaded.
# read from ../data\fever_train.jsonl
# length:  145327
# 100%|██████████| 145327/145327 [00:00<00:00, 1502239.31it/s]
# SUPPORT: 80035
# REFUTE: 29775
# NEI: 35517
# **********
#
# read from ../data\fever_validation.jsonl
# 100%|██████████| 9985/9985 [00:00<00:00, 2002109.45it/s]
# 100%|██████████| 809/809 [00:00<00:00, 811380.19it/s]
# length:  9985
# SUPPORT: 3333
# REFUTE: 3333
# NEI: 3319
# **********
#
# read from ../data\scifact_train.jsonl
# length:  809
# SUPPORT: 332
# REFUTE: 173
# NEI: 304
# **********
#
# read from ../data\scifact_validation.jsonl
# 100%|██████████| 300/300 [00:00<00:00, 300882.64it/s]
# length:  300
# SUPPORT: 124
# REFUTE: 64
# NEI: 112
# **********
#
# read from ../data\vc_train.jsonl
#   0%|          | 0/248953 [00:00<?, ?it/s]length:  248953
# 100%|██████████| 248953/248953 [00:00<00:00, 2097636.49it/s]
# SUPPORT: 124864
# REFUTE: 71108
# NEI: 52981
# **********
#
# read from ../data\vc_validation.jsonl
# 100%|██████████| 34481/34481 [00:00<00:00, 2161080.01it/s]
# length:  34481
# SUPPORT: 17306
# REFUTE: 9907
# NEI: 7268
# **********
#
#
# Process finished with exit code 0

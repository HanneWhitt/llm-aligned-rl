import json
import numpy as np
from matplotlib import pyplot as plt
import os
from LLM_judgement_request import make_jsonl_batch_file, submit_batch_file


SAVE_FOLDER = '../trajectory_dataset_3/'
OUTFOLDER = '../trajectory_dataset_3_jsonl/'
n_chunks = 25

chunk_length = int(10000/n_chunks)
episodes_chunks = [[f'episode_{ep}' for ep in range(x, x+chunk_length)] for x in range(0, 10000, chunk_length)]


print(len(episodes_chunks))

start_ep = 0
finish_ep = chunk_length - 1


batch_ids = []


for chunk in episodes_chunks:

    json_files_list = []
    req_ids_list = []

    for ep_name in chunk:
        ep_json = f'{SAVE_FOLDER}{ep_name}/{ep_name}.json'
        json_files_list.append(ep_json)
        req_ids_list.append(ep_name)

    output_file = f'{OUTFOLDER}{start_ep}_to_{finish_ep}.jsonl'
    # make_jsonl_batch_file(json_files_list, output_file, req_ids_list, image_selection='all')

    if start_ep != 0:
        batch_id = submit_batch_file(output_file)
        batch_ids.append(batch_id)
        print(batch_ids)

    start_ep += chunk_length
    finish_ep += chunk_length

np.savetxt("../batch_output/batch_ids.csv", batch_ids, delimiter=",")
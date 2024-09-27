from LLM_judgement_request import read_batch_output_file, get_info_from_batch_element, load_prompt, record_feedback
import json
import os
import time


SAVE_FOLDER = '../trajectory_dataset_3/'
batch_folder = '../batch_output/'

batch_files = [batch_folder + x for x in os.listdir(batch_folder) if x.endswith('.jsonl')]

print(len(batch_files))

start = {}


for batch_file in batch_files:

    res = read_batch_output_file(batch_file)

    for elem in res:

        ep_name, judgement, tokens_used = get_info_from_batch_element(elem)

        print(ep_name)

        ep_json = f'{SAVE_FOLDER}{ep_name}/{ep_name}.json'

        prompt_used = load_prompt()

        record_feedback(ep_json, judgement, 'all', prompt_used, tokens_used, batch_file, overwrite=True)




# id, judgement, tokens_used = get_info_from_batch_element(res[0])

# if id in start:
#     print('\nREPEATED')
#     print(id)
#     print(batch_file)
#     print(start[id])
#     print('\n')
# else:
#     start[id] = batch_file
#     print(id)




# result = get_batch_job(batch_id)

# if not os.path.isfile(results_file):

#     print('\nSTATUS - ', result.status)
#     failed += 1

#     input_file_id = result.input_file_id

#     batch_job = client.batches.create(
#         input_file_id=input_file_id,
#         endpoint="/v1/chat/completions",
#         completion_window="24h"
#     )
#     print(f'Batch job submitted. Job ID: \n {batch_job.id}\n')

#     new_batch_names.append(batch_job.id)

#     print('\nNEW BATCH NAMES')
#     print(new_batch_names)
    

#     time.sleep(180)



# res = read_batch_output_file(results_file)
# res = [get_info_from_batch_element(elem) for elem in res]

# for ep_name, judgement, tokens_used in res:
#     ep_json = f'{SAVE_FOLDER}{ep_name}/{ep_name}.json'
#     with open(ep_json) as f:
#         json_content = json.load(f)
#     true_answer = json_content['cat_squashed']
#     llm_response = judgement['question_3']
#     print(ep_name, llm_response, true_answer)
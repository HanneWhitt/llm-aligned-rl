from LLM_judgement_request import make_jsonl_batch_file, read_batch_output_file, get_info_from_batch_element
import json

SAVE_FOLDER = '../trajectory_dataset_3/'
results_file = '../batch_output/batch_66f5bead6a1481909b8eda85d400d324.jsonl'

res = read_batch_output_file(results_file)
res = [get_info_from_batch_element(elem) for elem in res]


# for state in ('alive', 'dead'):
#     for n_images in [5, 10, 20]:

for ep_name, judgement, tokens_used in res:
    ep_json = f'{SAVE_FOLDER}{ep_name}/{ep_name}.json'
    with open(ep_json) as f:
        json_content = json.load(f)
    true_answer = json_content['cat_squashed']
    llm_response = judgement['question_3']
    print(ep_name, llm_response, true_answer)
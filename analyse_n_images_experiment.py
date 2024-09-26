from LLM_judgement_request import make_jsonl_batch_file, read_batch_output_file, get_info_from_batch_element
import re

results_file = '../batch_output/batch_66f58cda89588190ad848577f53a9bac.jsonl'

res = read_batch_output_file(results_file)
res = [get_info_from_batch_element(elem) for elem in res]


# for state in ('alive', 'dead'):
#     for n_images in [5, 10, 20]:

patt = re.compile('([a-z]+)_([0-9]+)_all_episode_([0-9]+)')

for id, judgement, tokens_used in res:
    r = re.findall(patt, id)[0]
    llm_response = judgement['question_3']
    print(r, llm_response)
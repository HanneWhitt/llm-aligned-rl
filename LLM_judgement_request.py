from pydantic import BaseModel
from openai import OpenAI
import base64
import json
import time
from datetime import datetime
from ast import literal_eval
from pathlib import Path


class Judgement(BaseModel):
    question_1: str
    question_2: str
    question_3: bool


client = OpenAI()


def LLM_judgement_request(text_prompt, image_paths):
    start = time.time()
    content = [text_element(text_prompt)] + [image_element(image) for image in image_paths]
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        response_format=Judgement,
    )
    judgement = completion.choices[0].message.parsed
    request_time = time.time() - start
    print('Request returned in {:.1f}s'.format(request_time))
    return judgement, completion


# Needed to use batch API
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "Judgement",
        "schema": {
            "type": "object",
            "properties": {
                "question_1": {
                    "type": "string"
                },
                "question_2": {
                    "type": "string"
                },
                "question_3": {
                    "type": "boolean"
                }
            },
            "required": [
                "question_1",
                "question_2",
                "question_3"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}


def LLM_judgement_request_json_schema(text_prompt, image_paths):
    start = time.time()
    content = [text_element(text_prompt)] + [image_element(image) for image in image_paths]
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        response_format=response_format,
        temperature=0.0
    )
    judgement = json.loads(completion.choices[0].message.content)
    request_time = time.time() - start
    print('Request returned in {:.1f}s'.format(request_time))
    return judgement, completion


def LLM_judgement_request_batch_line(request_id, text_prompt, image_paths):
    content = [text_element(text_prompt)] + [image_element(image) for image in image_paths]
    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini-2024-07-18",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "response_format": response_format,
            "temperature": 0.0
        }
    }


def load_prompt(prompt_file='prompt.txt'):
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    return prompt


def select_image_files(episode_json, steps):

    with open(episode_json) as f:
        episode = json.load(f)
    trajectory = episode['trajectory']
    T = len(trajectory)

    if steps == 'all':
        steps_to_load = list(range(T))
    elif steps == 'first_and_last':
        steps_to_load = [0, T-1]
    elif isinstance(steps, int):
        steps_to_load = [x*(T-1)//(steps-1) for x in range(steps)]
    else:
        raise ValueError("steps arg must be 'all', 'first_and_last' or an int")
    
    image_files = [trajectory[st]['image_file'] for st in steps_to_load]

    return image_files


def text_element(text):
   return {
        "type": "text",
        "text": text
    }


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def image_element(image_path):
    base64_image = encode_image(image_path)
    element = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_image}",
            "detail": "low"
        }
    }
    return element


def record_feedback(json_file, judgement, steps_used, prompt_used, tokens_spent, from_batch_file=None, overwrite=False):
    is_dict = isinstance(judgement, dict)
    feedback = {
        'model_used': 'gpt-4o-mini-2024-07-18',
        'recorded_date': str(datetime.now()),
        'steps_used': steps_used,
        'tokens_spent': tokens_spent,
        'prompt_used': prompt_used,
        'question_1': judgement['question_1'] if is_dict else judgement.question_1,
        'question_2': judgement['question_2'] if is_dict else judgement.question_1,
        'question_3': judgement['question_3'] if is_dict else judgement.question_1,
        'from_batch_file': from_batch_file
    }
    with open(json_file) as f:
        json_content = json.load(f)
    if 'feedback' not in json_content:
        json_content = {'feedback': [], **json_content}
    if overwrite:
        json_content['feedback'] = [feedback]
    else:
        json_content['feedback'].append(feedback)
    with open(json_file, 'w') as f:
        json.dump(json_content, f)
    print('Feedback saved to ', json_file)



def make_jsonl_batch_file(json_files, output_file, req_ids=None, prompt='prompt.txt', image_selection='all'):    
    if prompt.endswith('.txt'):
        prompt = load_prompt(prompt)
    if req_ids is None:
        req_ids = [Path(ep_json).stem for ep_json in json_files]
    else:
        assert len(req_ids) == len(json_files)
    with open(output_file, 'w') as file:
        for i, ep_json in enumerate(json_files):
            req_id = req_ids[i]
            image_files = select_image_files(ep_json, image_selection)
            line = LLM_judgement_request_batch_line(req_id, prompt, image_files)
            file.write(json.dumps(line) + '\n')
    print('Requests written to ', output_file)
    return output_file


def submit_batch_file(jsonl_batch_file):
    batch_file = client.files.create(
        file=open(jsonl_batch_file, "rb"),
        purpose="batch"
    )
    print('Batch job uploaded')
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f'Batch job submitted. Job ID: \n {batch_job.id}\n')
    status = client.batches.retrieve(batch_job.id)
    print(status)
    return batch_job.id


def get_batch_job(batch_id, outfile=None):
    status = client.batches.retrieve(batch_id)
    print(f'Checking batch {batch_id}:\n\nStatus: {status.status}\n')
    print(status)
    if status.status == 'completed':
        print('COMPLETED - donwloading...')
        result_file_id = status.output_file_id
        result = client.files.content(result_file_id).content
        if outfile == None:
            outfile = f'../batch_output/{status.id}.jsonl'
        with open(outfile, 'wb') as file:
            file.write(result)
        print('Saved to ', outfile)
        return result
    return status


def read_batch_output_file(savefile):
    results = []
    with open(savefile, 'r') as file:
        for line in file:
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip())
            results.append(json_object)
    return results


#json_file, judgement, steps_used, prompt_used, tokens_spent
def get_info_from_batch_element(element):
    id = element['custom_id']
    judgement = json.loads(element['response']['body']['choices'][0]['message']['content'])
    tokens_used = element['response']['body']['usage']['total_tokens']
    return id, judgement, tokens_used

if __name__ == '__main__':

    ep_no = 9
    ep_json = f'../trajectory_dataset/episode_{ep_no}/episode_{ep_no}.json'

    make_jsonl_batch_file([ep_json], 'test_jsonl.jsonl')

    submit_batch_file('test_jsonl.jsonl')

    

    # prompt = load_prompt()
    # print(prompt)

    # image_files = select_image_files(ep_json, 'all')

    # for imf in image_files:
    #     print(imf)



    # judgement, completion = LLM_judgement_request_json_schema(prompt, image_files)
    # tokens = completion.usage.total_tokens

    # print(judgement)

    #print(json.dumps(judgement, indent=4))

    # record_feedback(ep_json, judgement, 'all', prompt, tokens)

    
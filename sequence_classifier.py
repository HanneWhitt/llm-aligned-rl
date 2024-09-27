import os


def get_judgement(ep_json):
    return ep_json['feedback'][0]['question_3']


def load_example(ep_name):
    


if __name__ == '__main__':
    SAVE_FOLDER = '../trajectory_dataset_3/'
    episodes = os.listdir(SAVE_FOLDER)

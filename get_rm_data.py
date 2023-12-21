import regex
import jsonlines
from tqdm import tqdm
from string import punctuation
from mcts_rl.configs.constants import COT_INSTRUCTIONS, PROMPT_BEGIN, PROMPT_ASSISTANT, PROMPT_USER

def load_jsonl(fname):
    with jsonlines.open(fname, mode='r') as reader:
        data = [l for l in reader]
    return data

def dump_jsonl(data, fname):
    with jsonlines.open(fname, mode='w') as writer:
        writer.write_all(data)

def eval_accu(prediction, option, answer):
    if prediction is None:
        return 1
    
    raw_prediction = prediction
    format_flag = False
    
    if 'QUESTION:' in prediction:
        prediction = prediction.split('QUESTION:')[0]
    
    if ' answer is' in prediction:
        prediction = prediction.strip().split(' answer is')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'Overall, ' in prediction:
        prediction = prediction.strip().split('Overall, ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'Answer: ' in prediction:
        prediction = prediction.strip().split('Answer: ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'Therefore, ' in prediction:
        prediction = prediction.strip().split('Therefore, ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'Thus, ' in prediction:
        prediction = prediction.strip().split('Thus, ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'So, ' in prediction:
        prediction = prediction.strip().split('So, ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'So ' in prediction:
        prediction = prediction.strip().split('So ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif '\n' in prediction:
        prediction = prediction.strip().split('\n')[-1].strip().strip(punctuation).strip()
    
    options = regex.findall(r'\([A-Z1-9]\)|[A-Z1-9]\)', prediction)
    options_backup = regex.findall(r'[A-Z1-9]', prediction)
    if options:
        options = [x for i, x in enumerate(options) if x not in options[i+1:]]
        prediction = options[-1].strip(punctuation)
    elif options_backup and format_flag:
        options = [x for i, x in enumerate(options_backup) if x not in options_backup[i+1:]]
        prediction = options[-1].strip(punctuation)
    else:
        options = regex.findall(r'\([A-Z1-9]\)|[A-Z1-9]\)', raw_prediction)
        if options:
            options = [x for i, x in enumerate(options) if x not in options[i+1:]]
            prediction = options[-1].strip(punctuation)
    # return options[-1].strip(punctuation) == option.strip(punctuation) if options else False
    # if options:
    #     options = [x.strip(punctuation) for x in options]
    #     return options.count(option.strip(punctuation)) / len(options)
    # else:
    #     return 0
    
    if prediction == option.strip(punctuation) and len(options) == 1 and format_flag:
        return 1
    elif option.strip(punctuation) in [x.strip(punctuation) for x in options]:
        return 1 / len(options)
        # return options[0].strip(punctuation) == option.strip(punctuation)
        # return int(len(options) == 1 and prediction == option.strip(punctuation))
    elif answer in prediction:
        return 0
    return 0

def ranked_predictions(fname):
    rawdata = load_jsonl(fname)
    data = []
    for dt in tqdm(rawdata):
        prompt = dt['prompt'][0].strip().replace(PROMPT_BEGIN, '').replace(PROMPT_USER, '').replace(PROMPT_ASSISTANT, '').strip()
        generations = []
        for gen in dt['generated']:
            gen = gen.strip()
            _eval = eval_accu(gen, dt['answer'], dt['answer_content'])
            generations.append((gen, _eval))
        generations.sort(key=lambda x: -x[1])
        better, worse = generations[0], generations[-1]
        if better[-1] <= worse[-1]:
            continue
        left, right = 0, len(generations) - 1
        while left < right:
            better, worse = generations[left], generations[right]
            left += 1
            right -= 1
            if better[-1] <= worse[-1]:
                break
            data.append({
                'prompt': prompt,
                'response_0': better[0],
                'response_1': worse[0],
                'is_response_0_correct': better[-1] == 1,
                'is_response_1_correct': worse[-1] == 1,
            })
    return data

if __name__ == '__main__':
    filepath = '/mnt/data/yuxi/mcts-rl/predictions/mcq/sqa/train-mistral-mj10-cot.jsonl'
    dump_jsonl(
        ranked_predictions(filepath),
        '/mnt/data/yuxi/reward-model/sqa-pairs-all.jsonl',
    )
    
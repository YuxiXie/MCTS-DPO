import regex
import jsonlines
from collections import defaultdict
from string import punctuation

def load_jsonl(fname):
    with jsonlines.open(fname, mode='r') as reader:
        data = [l for l in reader]
    return data

def _extract_answer(gen):
    raw_gen = gen
    format_flag = False
    if 'QUESTION:' in gen:
        gen = gen.split('QUESTION:')[0]
    if ' answer is' in gen:
        gen = gen.strip().split(' answer is')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'Overall, ' in gen:
        gen = gen.strip().split('Overall, ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'Answer: ' in gen:
        gen = gen.strip().split('Answer: ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'Therefore, ' in gen:
        gen = gen.strip().split('Therefore, ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'Thus, ' in gen:
        gen = gen.strip().split('Thus, ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'So, ' in gen:
        gen = gen.strip().split('So, ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif 'So ' in gen:
        gen = gen.strip().split('So ')[-1].strip().strip(punctuation).strip()
        format_flag = True
    elif '\n' in gen:
        gen = gen.strip().split('\n')[-1].strip().strip(punctuation).strip()
    
    options = regex.findall(r'\([A-Z1-9]\)|[A-Z1-9]\)', gen)
    options_backup = regex.findall(r'[A-Z1-9]', gen)
    if options:
        options = [x for i, x in enumerate(options) if x not in options[i+1:]]
        prediction = options[-1].strip(punctuation)
    elif options_backup and format_flag:
        options = [x for i, x in enumerate(options_backup) if x not in options_backup[i+1:]]
        prediction = options[-1].strip(punctuation)
    else:
        options = regex.findall(r'\([A-Z1-9]\)|[A-Z1-9]\)', raw_gen)
        if options:
            options = [x for i, x in enumerate(options) if x not in options[i+1:]]
            prediction = options[-1].strip(punctuation)
        else:
            prediction = None
    return prediction

def eval_accu(prediction, option, answer, k=100):    
    if isinstance(prediction, str):
        return _extract_answer(prediction) == option.strip(punctuation)
    else:
        counter = defaultdict(int)
        for g in prediction[:k]:
            g = _extract_answer(g)
            if g is not None:
                counter[g] += 1
        return option.strip(punctuation) in counter
        try:
            return max(counter.items(), key=lambda x: x[1])[0] == option.strip(punctuation)
        except:
            return False

def load_raw_labels(dtype, valtest=False):
    if valtest:
        raw_labels = load_jsonl(f'/home/users/nus/e0672129/scratch/csr/mcq_{dtype}_test.jsonl')
    else:
        raw_labels = load_jsonl(f'/home/users/nus/e0672129/scratch/csr/mcq_{dtype}_fulltest.jsonl')
    raw_labels_dict = {}
    for i, dt in enumerate(raw_labels):
        qu = (dt['question'].strip(), dt['answer'])
        raw_labels_dict[qu] = dt['label']
    return raw_labels_dict

from mcts_rl.configs.constants import COT_INSTRUCTIONS, PROMPT_BEGIN, PROMPT_ASSISTANT, PROMPT_USER, SQA_PROMPT

SQA_PROMPT = SQA_PROMPT.replace('</s>\n\n', ' ').replace('</s>', '').strip()

def mcq_extract_pred_result(raw_preds, dtype='default'):
    predictions, lens = {}, []
    if len(raw_preds) > 100:
        raw_preds = [raw_preds]
    for raw_pred in raw_preds:
        for dt in raw_pred:
            prompt = dt['prompt'][0].strip().replace(SQA_PROMPT, '').strip().replace(PROMPT_BEGIN, '').replace(PROMPT_USER, '').split(PROMPT_ASSISTANT)[0].strip()
            if dtype == 'mcts':
                generated = dt['generated'][-1][-1] if len(dt['generated']) else None
            else:
                generated = dt['generated'][0] if len(dt['generated']) == 0 else dt['generated']
            lens.append(len(dt['generated']))
            gt_answer = (dt['answer'], dt['answer_content'],)
            if prompt not in predictions:
                predictions[prompt] = {'pred': generated, 'gt_answer': gt_answer}
            elif isinstance(predictions[prompt]['pred'], list):
                predictions[prompt]['pred'] += generated
    return predictions

def mcq_visualize_pred_result(predictions, N=int(1e5), dtype='csr', show_split=False, valtest=False, k=100):
    raw_labels_dict = load_raw_labels(dtype, valtest=valtest)
    accu = []
    tsk_accu = None
    if dtype == 'sqa':
        tsk_accu = {x:[] for x in ['openbook', 'arc_easy', 'arc_hard', 'ai2s_ele', 'ai2s_mid']}
    elif dtype == 'csr':
        tsk_accu = {x:[] for x in ['csqa', 'siqa', 'piqa']}
    for prompt, gens in predictions.items():
        sft_gen = gens['pred']
        _eval = eval_accu(sft_gen, gens['gt_answer'][0], gens['gt_answer'][1], k=k)
        tsk = prompt.replace('QUESTION:', '').strip().split(PROMPT_ASSISTANT)[0].strip()
        tsk = (tsk, gens['gt_answer'][0])
        if tsk not in raw_labels_dict:
            continue
        if tsk_accu is not None and raw_labels_dict[tsk] not in tsk_accu:
            continue
        accu.append(_eval)
        if tsk_accu is not None:
            tsk_accu[raw_labels_dict[tsk]].append(_eval)
        if len(accu) >= N:
            break

    print('* all', sum(accu)/max(1, len(accu)), '({})'.format(len(accu)))
    if not show_split or tsk_accu is None:
        return
    for k, v in tsk_accu.items():
        print(k, sum(v)/max(1, len(v)), '({})'.format(len(v)))

N = 3168
dtype = 'sqa'
# mcq_visualize_pred_result(
#     mcq_extract_pred_result(
#         [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sc20-arithmo.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/mcq/predictions/base-sc.jsonl')],
#         dtype=dtype,
#     ),
#     dtype=dtype,
#     N=N,
#     show_split=True,
# )

dtype = 'sciq'
# mcq_visualize_pred_result(
#     mcq_extract_pred_result(
#         [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sft.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s1024.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s1024.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s2048.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s2048.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s2432.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s3072.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s5888.jsonl'),
#         ],
#         dtype=dtype,
#     ),
#     dtype=dtype,
#     N=N,
#     valtest=True,
# )
# mcq_visualize_pred_result(
#     mcq_extract_pred_result(
#         [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sft.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s1024.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s1024.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s2048.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s2048.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s2432.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s3072.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s5888.jsonl'),
#          ],
#         dtype=dtype,
#     ),
#     dtype=dtype,
#     N=N,
#     valtest=True,
# )
# mcq_visualize_pred_result(
#     mcq_extract_pred_result(
#         [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sft.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s1024.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s1024.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s2048.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s2048.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s2432.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s3072.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s5888.jsonl'),
#          ],
#         dtype=dtype,
#     ),
#     dtype=dtype,
#     N=N,
#     valtest=True,
# )
# mcq_visualize_pred_result(
#     mcq_extract_pred_result(
#         [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sft.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s1024.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s1024.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s2048.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s2048.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s2432.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s3072.jsonl'),
#         #  load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s5888.jsonl'),
#         ],
#         dtype=dtype,
#     ),
#     dtype=dtype,
#     N=N,
#     valtest=True,
# )
# mcq_visualize_pred_result(
#     mcq_extract_pred_result(
#         [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sft.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s1024.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s1024.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s2048.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s2048.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s2432.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-ipo-mistral-online-mcts-s3072.jsonl'),
#          load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-mistral-online-mcts-s5888.jsonl'),],
#         dtype=dtype,
#     ),
#     dtype=dtype,
#     N=N,
#     valtest=True,
# )


mcq_visualize_pred_result(
    mcq_extract_pred_result(
        [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft2.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft3.jsonl'),],
        dtype=dtype,
    ),
    dtype=dtype,
    N=N,
    valtest=True,
    k = 5
)
mcq_visualize_pred_result(
    mcq_extract_pred_result(
        [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft2.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft3.jsonl'),],
        dtype=dtype,
    ),
    dtype=dtype,
    N=N,
    valtest=True,
    k = 6
)
mcq_visualize_pred_result(
    mcq_extract_pred_result(
        [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft2.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft3.jsonl'),],
        dtype=dtype,
    ),
    dtype=dtype,
    N=N,
    valtest=True,
    k = 12
)
mcq_visualize_pred_result(
    mcq_extract_pred_result(
        [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft2.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft3.jsonl'),],
        dtype=dtype,
    ),
    dtype=dtype,
    N=N,
    valtest=True,
    k = 16
)
mcq_visualize_pred_result(
    mcq_extract_pred_result(
        [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft2.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft3.jsonl'),],
        dtype=dtype,
    ),
    dtype=dtype,
    N=N,
    valtest=True,
    k = 7
)
mcq_visualize_pred_result(
    mcq_extract_pred_result(
        [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft2.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft3.jsonl'),],
        dtype=dtype,
    ),
    dtype=dtype,
    N=N,
    valtest=True,
    k = 8
)
mcq_visualize_pred_result(
    mcq_extract_pred_result(
        [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft2.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft3.jsonl'),],
        dtype=dtype,
    ),
    dtype=dtype,
    N=N,
    valtest=True,
    k = 9
)
mcq_visualize_pred_result(
    mcq_extract_pred_result(
        [load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft2.jsonl'),
         load_jsonl('/home/users/nus/e0672129/scratch/MCTS-DPO/outputs/experiments/sqa/predictions/sciq-sc-sft3.jsonl'),],
        dtype=dtype,
    ),
    dtype=dtype,
    N=N,
    valtest=True,
    k = 10
)


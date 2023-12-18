import regex
import jsonlines
from string import punctuation

def load_jsonl(fname):
    with jsonlines.open(fname, mode='r') as reader:
        data = [l for l in reader]
    return data

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
        # return 1 / len(options)
        return prediction == option.strip(punctuation)
    elif answer in prediction:
        return 0
    return 0

from mcts_rl.configs.constants import COT_INSTRUCTIONS, PROMPT_BEGIN, PROMPT_ASSISTANT, PROMPT_USER

def mcq_extract_pred_result(raw_pred, dtype='default'):
    predictions, lens = {}, []
    for dt in raw_pred:
        prompt = dt['prompt'][0].strip().replace(PROMPT_BEGIN, '').replace(PROMPT_USER, '').replace(PROMPT_ASSISTANT, '').strip()
        if dtype == 'mcts':
            generated = dt['generated'][-1][-1] if len(dt['generated']) else None
        else:
            generated = dt['generated'][0]
        lens.append(len(dt['generated']))
        gt_answer = (dt['answer'], dt['answer_content'],)
        if prompt in predictions: continue
        predictions[prompt] = {'pred': generated, 'gt_answer': gt_answer}

    print(max(lens), sum(lens)/len(lens))
    return predictions

def mcq_visualize_pred_result(predictions, N=1e5, dtype='csr'):
    raw_labels = load_jsonl(f'/home/users/nus/e0672129/scratch/csr/mcq_{dtype}_test.jsonl')
    raw_labels_dict = {}
    for i, dt in enumerate(raw_labels):
        qu = (dt['question'], dt['answer'])
        raw_labels_dict[qu] = dt['label']
    
    accu = []
    if dtype == 'sqa':
        tsk_accu = {x:[] for x in ['openbook', 'arc_easy', 'arc_hard', 'ai2s_ele', 'ai2s_mid']}
    elif dtype == 'csr':
        tsk_accu = {x:[] for x in ['winogrande', 'csqa', 'siqa', 'piqa']}
    for prompt, gens in predictions.items():
        sft_gen = gens['pred']
        _eval = eval_accu(sft_gen, gens['gt_answer'][0], gens['gt_answer'][1])
        accu.append(_eval)
        tsk = prompt.replace('QUESTION:', '').strip().replace("ANSWER: Let's think step by step.", '').strip().replace('\n\nANSWER:', '').strip()
        tsk = (tsk, gens['gt_answer'][0])
        if tsk not in raw_labels_dict:
            continue
        tsk_accu[raw_labels_dict[tsk]].append(_eval)
        if len(accu) >= N:
            break

    print('all', sum(accu)/max(1, len(accu)), '({})'.format(len(accu)))
    for k, v in tsk_accu.items():
        print(k, sum(v)/max(1, len(v)), '({})'.format(len(v)))

def mcq_show_rst(fname, dtype='csr'):
    mcq_visualize_pred_result(
        mcq_extract_pred_result(
            load_jsonl(fname),
            dtype=dtype,
        ),
        dtype=dtype
    )


import re

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string

def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

ANSWER_INDICATOR = 'he answer is'

def extract_answer(pred_str):
    if 'USER:' in pred_str:
        pred_str = pred_str.split('USER:')[0]
    if 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans) == 0:
            return ""
        elif (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        pred=a
    elif (ANSWER_INDICATOR in pred_str):
        pred = pred_str.split(ANSWER_INDICATOR)[-1].strip()
    else: # use the last number
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str.replace(",", ""))
        if(len(pred) >= 1):
            pred = pred[-1]
        else: pred = ''
    
    # multiple line
    pred = pred.split("\n")[0]
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred

from math import isclose
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex

def is_digit(s):
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False

def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except:
                pass
        return s
    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a-b) == 0:
            return True
    except:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except:
        pass
    return False

def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)

def math_equal(prediction, reference,
                include_percentage: bool = True,
                is_close: bool = True,
                timeout: bool = False,
                ) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    try: # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = float(str(prediction).replace(",", ""))
            reference = float(str(reference).replace(",", ""))
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, rel_tol=1e-4):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or \
        (prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ['{', "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (prediction.startswith("[") and prediction.endswith("]")) and (reference.startswith("[") and reference.endswith("]")) or \
        (prediction.startswith("(") and prediction.endswith(")")) and (reference.startswith("(") and reference.endswith(")")):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all([math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close) for i in range(len(pred_parts))]):
                return True

    # symbolic equal with sympy
    if symbolic_equal(prediction, reference):
        return True

    return False

import jsonlines
from string import punctuation

def load_jsonl(fname):
    with jsonlines.open(fname, mode='r') as reader:
        data = [l for l in reader]
    return data

from mcts_rl.configs.constants import COT_INSTRUCTIONS, PROMPT_BEGIN, PROMPT_ASSISTANT, PROMPT_USER

def extract_pred_result(raw_pred, dtype='default'):
    predictions, lens = {}, []
    for dt in raw_pred:
        prompt = dt['prompt'][0].strip().replace(PROMPT_BEGIN, '').replace(PROMPT_USER, '').replace(PROMPT_ASSISTANT, '').strip()
        if dtype == 'mcts':
            generated = dt['generated'][-1][-1] if len(dt['generated']) else None
        else:
            generated = dt['generated'][0]
        lens.append(len(dt['generated']))
        gt_answer = (dt['answer'], dt['answer_content'],)
        if prompt in predictions: continue
        predictions[prompt] = {'pred': generated, 'gt_answer': gt_answer}

    print(max(lens), sum(lens)/len(lens))
    return predictions

def visualize_pred_result(predictions, N=int(1e5), dtype='csr'):
    accu = []
    for prompt, gens in predictions.items():
        sft_gen = gens['pred']
        accu.append(sft_gen is None or math_equal(extract_answer(sft_gen), gens['gt_answer'][0]))

    print('all', sum(accu[:N])/max(1, len(accu[:N])), '({})'.format(len(accu[:N])))

def show_rst(fname):
    visualize_pred_result(
        extract_pred_result(
            load_jsonl(fname)
        )
    )

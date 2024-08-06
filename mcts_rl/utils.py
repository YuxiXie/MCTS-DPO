# Adapted from: https://github.com/PKU-Alignment/safe-rlhf/blob/main/safe_rlhf/utils.py

"""Miscellaneous utilities."""

from __future__ import annotations

import dataclasses
import os
import sys
import regex
import random
import threading
import func_timeout
from io import StringIO
from tqdm import tqdm
from contextlib import redirect_stdout
from collections import OrderedDict
from typing import Any, Callable, Generator, TypeVar, cast
from typing_extensions import TypeAlias  # Python 3.10+

from nltk.tokenize import sent_tokenize

import numpy as np
import optree
from string import punctuation
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from optree.typing import PyTreeTypeVar
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import ModelOutput
from transformers.tokenization_utils import BatchEncoding, PaddingStrategy, TruncationStrategy

from mcts_rl.configs.constants import PROMPT_ASSISTANT, IGNORE_INDEX


__all__ = [
    'seed_everything',
    'str2bool',
    'to_device',
    'batch_retokenize',
    'is_same_tokenizer',
    'is_main_process',
    'get_all_reduce_mean',
    'get_all_reduce_sum',
    'get_optimizer_grouped_parameters',
    'gather_log_probabilities',
]


TensorTree: TypeAlias = PyTreeTypeVar('TensorTree', torch.Tensor)
Func = TypeVar('Func', bound=Callable[..., Any])


def seed_everything(seed: int) -> None:
    """Set global random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(string: str) -> bool:
    """Convert a string literal to a boolean value."""
    if string.lower() in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if string.lower() in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    return bool(string)


def get_subclasses(cls: type, memo: set[type] | None = None) -> Generator[type, None, None]:
    """Get all subclasses of a class recursively."""
    if memo is None:
        memo = set()

    for subclass in cls.__subclasses__():
        if subclass in memo:
            continue

        memo.add(subclass)
        yield subclass
        yield from get_subclasses(subclass, memo=memo)


__PYTREE_INITIALIZED = False
__PYTREE_REGISTRY_LOCK = threading.Lock()


def __initialize_pytree_registry_once() -> None:
    # pylint: disable-next=import-outside-toplevel,unused-import
    from mcts_rl.models.score_model import ScoreModelOutput  # noqa: F401

    global __PYTREE_INITIALIZED  # pylint: disable=global-statement
    if __PYTREE_INITIALIZED:
        return

    with __PYTREE_REGISTRY_LOCK:
        if __PYTREE_INITIALIZED:
            return

        optree.register_pytree_node(
            BatchEncoding,
            lambda batch_encoding: (
                [batch_encoding.data],
                {'encoding': batch_encoding.encodings, 'n_sequences': batch_encoding.n_sequences},
            ),
            lambda metadata, children: BatchEncoding(children[0], **metadata),
            namespace='mcts_rl',
        )
        optree.register_pytree_node(
            ModelOutput,
            lambda model_output: (model_output.values(), model_output.keys(), model_output.keys()),
            lambda keys, values: ModelOutput(OrderedDict(zip(keys, values))),
            namespace='mcts_rl',
        )

        for model_output_class in filter(dataclasses.is_dataclass, get_subclasses(ModelOutput)):
            optree.register_pytree_node(
                model_output_class,
                lambda model_output: ([dataclasses.asdict(model_output)], type(model_output)),
                lambda metadata, children: metadata(**children[0]),
                namespace='mcts_rl',
            )

        __PYTREE_INITIALIZED = True


def to_device(batch: TensorTree, device: torch.device | str | int | None) -> TensorTree:
    """Move a batch of tensors to a device."""
    if not __PYTREE_INITIALIZED:
        __initialize_pytree_registry_once()
    if device is None:
        return batch
    return optree.tree_map(lambda x: x if isinstance(x, str) else x.to(device), batch, namespace='mcts_rl')


def batch_retokenize(
    input_ids: torch.LongTensor,
    src_tokenizer: PreTrainedTokenizerBase,
    dest_tokenizer: PreTrainedTokenizerBase,
    *,
    padding: bool | str | PaddingStrategy = PaddingStrategy.LONGEST,
    truncation: bool | str | TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    skip_special_tokens: bool = True,
    device: torch.device | str | int | None = None,
) -> BatchEncoding:
    """Re-tokenize a batch of input ids from one tokenizer to another."""
    output = dest_tokenizer(
        [
            text + dest_tokenizer.eos_token
            for text in src_tokenizer.batch_decode(
                input_ids,
                skip_special_tokens=skip_special_tokens,
            )
        ],
        padding=padding,
        truncation=truncation,
        return_tensors='pt',
    )
    if device is not None:
        output = to_device(output, device)
    return output


def is_same_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    other_tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """Check if two tokenizers are the same."""
    return tokenizer is other_tokenizer or (
        tokenizer.__class__ == other_tokenizer.__class__
        and tokenizer.get_vocab() == other_tokenizer.get_vocab()
    )


def is_main_process() -> bool:
    """Check if the current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def rank_zero_only(func: Func) -> Func:
    """Decorator to make a function only run on the main process."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function for the decorator."""
        if is_main_process():
            return func(*args, **kwargs)
        return None

    return cast(Func, wrapper)


def get_all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Perform all-reduce operation on a tensor cross all ranks and return the mean."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def get_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Perform all-reduce operation on a tensor cross all ranks and return the sum."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def get_all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    """Perform all-reduce operation on a tensor cross all ranks and return the max."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def get_optimizer_grouped_parameters(
    module: nn.Module,
    weight_decay: float,
    no_decay_name_set: set[str] | None = None,
) -> list[dict[str, list[nn.Parameter] | float]]:
    """Get parameter groups with customized weight decay value."""
    if no_decay_name_set is None:
        no_decay_name_set = {'bias', 'LayerNorm.weight'}
    no_decay_name_set = set(map(str.lower, no_decay_name_set))

    named_parameters = [
        (name.lower(), param) for name, param in module.named_parameters() if param.requires_grad
    ]

    return [
        {
            'params': [
                param
                for name, param in named_parameters
                if not any(no_decay_name in name for no_decay_name in no_decay_name_set)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                param
                for name, param in named_parameters
                if any(no_decay_name in name for no_decay_name in no_decay_name_set)
            ],
            'weight_decay': 0.0,
        },
    ]


def gather_log_probabilities(logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
    """Gather log probabilities of the given labels from the logits."""
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(dim=-1))
    return log_probs_labels.squeeze(dim=-1)


def split_prompt_response(
    texts: list[str],
    split_token: str = PROMPT_ASSISTANT,
) -> tuple[list[str], list[str]]:
    """Split prompt-response pairs into prompts and responses."""

    def split_fn(text: str) -> tuple[str, str]:
        """Split a prompt-response pair into prompt and response."""
        prompt, partition, response = text.rpartition(split_token)
        assert prompt and partition and response, f'invalid text: {text}'
        return prompt + partition, response

    return tuple(map(list, zip(*map(split_fn, texts))))


def check_diversity(values):
    for scores in values[::-1]:
        if scores[-1] - scores[0] > .3 and scores[-1] > 0 and scores[0] < 1:
            return True
    return False


def check_available(rl_batch, max_tokens=512, eos_token_id=2, to_filter=False):
    if len(rl_batch) < 3:
        return False
    if 'input_ids' in rl_batch:
        return len(rl_batch['input_ids']) >= 2 and eos_token_id in rl_batch['input_ids'][-1].tolist()
    if to_filter:
        # return check_diversity(rl_batch['init_value_list'])
        if not len(rl_batch['input_ids_list']):
            return False
        if rl_batch['input_ids_list'][-1].size(-1) >= max_tokens and eos_token_id not in rl_batch['input_ids_list'][-1][-1]:
            return False
        return rl_batch.get('prediction', [False])[-1] > 0.8 # or random.random() < .1
    input_ids_list = rl_batch['input_ids_list']
    counts = [
        input_ids.size(0) >= 2 and input_ids.size(-1) <= max_tokens and (input_ids[0] != input_ids[1]).nonzero().size(0)
        for input_ids in input_ids_list
    ]
    return any(counts)


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
    _string = re.sub(r" [\w\.\s]+$", "", string).strip()
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
    if string[-1] == ".":
        string = string[:-1]

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

ANSWER_INDICATOR = ' answer is'


def safe_execute(code_string: str, maxtime=1):
    def execute(x):
        try:
            f = StringIO()
            with redirect_stdout(f):
                exec(f'from math import *\n{x}')
            return f.getvalue().strip()
        except Exception as e:
            return 'error'
    try:
        ans = func_timeout.func_timeout(maxtime, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = 'timeerror'
    return ans

def extract_answer(pred_str, use_code=False):
    if use_code:
        return safe_execute(pred_str.strip())
    
    if 'USER:' in pred_str:
        pred_str = pred_str.split('USER:')[0]
    if (ANSWER_INDICATOR in pred_str):
        pred = pred_str.split(ANSWER_INDICATOR)[-1].strip()
    elif 'boxed' in pred_str:
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

def csr_equal(prediction, reference):
    raw_prediction = prediction
    option, answer = reference
    format_flag = False
    
    if 'QUESTION:' in prediction:
        prediction = prediction.split('QUESTION:')[0]
    if ' answer is' in prediction:
        prediction = prediction.strip().split(' answer is')[-1].strip()
        format_flag = True
    elif 'Overall, ' in prediction:
        prediction = prediction.strip().split('Overall, ')[-1].strip()
        format_flag = True
    elif 'Answer: ' in prediction:
        prediction = prediction.strip().split('Answer: ')[-1].strip()
        format_flag = True
    elif 'In summary, ' in prediction:
        prediction = prediction.strip().split('In summary, ')[-1].strip()
        format_flag = True
    elif 'In conclusion, ' in prediction:
        prediction = prediction.strip().split('In conclusion, ')[-1].strip()
        format_flag = True
    elif 'Therefore, ' in prediction:
        prediction = prediction.strip().split('Therefore, ')[-1].strip()
        format_flag = True
    elif 'Thus, ' in prediction:
        prediction = prediction.strip().split('Thus, ')[-1].strip()
        format_flag = True
    elif 'So, ' in prediction:
        prediction = prediction.strip().split('So, ')[-1].strip()
        format_flag = True
    elif 'So ' in prediction:
        prediction = prediction.strip().split('So ')[-1].strip()
        format_flag = True
    elif '\n' in prediction:
        prediction = regex.split(r'[\n]+', prediction.strip())[-1].strip()
    
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
    
    # sents = sent_tokenize(raw_prediction)    
    # if regex.match(f'The .*answer is .*[A-Z1-9]', raw_prediction.strip()) or \
    #     (len(sents) == 1 and regex.search('.* answer is .*[A-Z1-9]', raw_prediction)):
    #     # Simply returning the final results is not valid
    #     return 0 if prediction == option.strip(punctuation) else -1    
    
    if prediction == option.strip(punctuation) and len(options) == 1 and format_flag:
        # Correct answer & readable format
        return 1
    elif option.strip(punctuation) in [x.strip(punctuation) for x in options]:
        # Got more than one answers or not in the readable format
        return .9 / len(options)
    elif answer in prediction:
        # Only provide textual answer
        return 0
    return -1


def get_choice_content(choice, question):
    xsplit = 'Answer Choices:'
    if 'answer choices:' in question:
        xsplit = 'answer choices:'
    elif 'Answer choices:' in question:
        xsplit = 'Answer choices:'
    choices = question.split(xsplit)[-1].strip()[1:].split(' (')
    for option in choices:
        o, c = option.split(')')[0], ')'.join(option.split(')')[1:])
        if o.strip().lower() == choice.strip().lower():
            return c.strip()


def get_math_data(rawdata, use_code=False):
    outdata = []
    for dt in tqdm(rawdata, desc='process math data', leave=False):
        question, answer = dt['question'], dt['answer']
        if 'The answer is' in answer and not use_code:
            final_answer = extract_answer(answer)
            if regex.match(r'\b[A-Za-z]\b', final_answer):
                if 'answer choices:' not in question.lower(): continue
            if 'answer choices:' in question.lower():
                if not regex.match(r'\b[A-Za-z]\b', final_answer): continue
            if answer.lower().startswith('the answer is'): continue
            outdata.append({
                'question': question, 'solution': answer,
                'answer': final_answer,
                'answer_content': get_choice_content(final_answer, question) if regex.match(r'\b[A-Za-z]\b', final_answer) else final_answer,
            })
        elif regex.search(r'print\(.+\)', answer) and use_code:
            final_answer = extract_answer(answer, use_code=True)
            if final_answer:
                outdata.append({
                    'question': question, 'solution': answer,
                    'answer': final_answer,
                    'answer_content': get_choice_content(final_answer, question) if regex.match(r'\b[A-Za-z]\b', final_answer) \
                        and 'answer choices:' in question.lower() else final_answer,
                })
    return outdata

def get_arithmo_data(mathqa_dict):
    return [random.choice(v) for v in mathqa_dict.values()]

def calculate_preference_confidence(qb, qw):
    return 1 / (1 + qw / max(1, qb))


def calculate_diversity_score(candidates):
    if candidates is None: return 0
    
    Q_values = [sample.Q for sample in candidates]
    variance = np.var(np.asarray(Q_values))
    gap = max(Q_values) - min(Q_values)
    # return gap if max(Q_values) > 0 else gap * 0.5
    
    visit_counts = [sample.N for sample in candidates]
    gap = max(visit_counts) - min(visit_counts)
    return gap


QUESTION_TOKEN_IDS = [35780, 2849, 25]

def get_final_qa_index(input_ids):
    question_indexes = input_ids.eq(QUESTION_TOKEN_IDS[-1]).nonzero()
    question_indexes = [_ for _ in question_indexes]
    question_index = question_indexes[-1] - (len(QUESTION_TOKEN_IDS) - 1)
    question_indexes = question_indexes[::-1]
    for idx in question_indexes:
        if input_ids[idx - (len(QUESTION_TOKEN_IDS) - 1): idx + 1].tolist() == QUESTION_TOKEN_IDS:
            question_index = idx - (len(QUESTION_TOKEN_IDS) - 1)
            break
    return question_index.item()


def list_to_dict(data):
    _dict = {}
    for dt in data:
        prompt = dt['question'] if 'question' in dt else dt['problem']
        if not prompt.strip(): continue
        answer = dt['solution'] if 'solution' in dt else dt['answer']
        if not answer.strip(): continue
        if prompt not in _dict:
            _dict[prompt] = []
        _dict[prompt].append(dt)
    return _dict


def pad_tensors(tensors, max_len=-1, pad_value=IGNORE_INDEX):
    tensors = [x for x in tensors]
    if max_len <= 0:
        max_len = max([len(x) for x in tensors])
    for i in range(len(tensors)):
        pad_len = max_len - len(tensors[i])
        tmp = torch.ones((pad_len,), dtype=torch.long, device=tensors[i].device)
        tensors[i] = torch.cat((tensors[i], tmp * pad_value), dim=-1).long()
    return torch.stack(tensors, dim=0)

import math
import pickle
import torch
import torch.distributed as dist
from bert_score.utils import greedy_cos_idf
from torch.nn.utils.rnn import pad_sequence


def get_embs_masks(seq_ids: list[list[int]], seq_embs: list[torch.Tensor], 
                   special_ids: list = [1, 2, 32000]):
    seq_ids = [ids[:len(embs)] for ids, embs in zip(seq_ids, seq_embs)]
    seq_embs = pad_sequence(seq_embs, batch_first=True, padding_value=2.0)
    lengths = torch.tensor([len(seq) for seq in seq_ids], dtype=torch.long)
    seq_masks = torch.arange(max(lengths), dtype=torch.long).expand(len(lengths), max(lengths))
    seq_masks = seq_masks < lengths.unsqueeze(1)
    seq_idfs = pad_sequence([
        torch.tensor([float(_id not in special_ids) for _id in seq])
    for seq in seq_ids], batch_first=True, padding_value=0.0)
    return seq_embs, seq_masks.to(seq_embs.device), seq_idfs.to(seq_embs.device)


def filter_with_similarity(candidates, tokenizer):
    sequences = [tokenizer.decode(x[0]) for x in candidates]
    special_ids = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]
    
    seq_ids, seq_embs = [c[0][:len(c[2])] for c in candidates], [c[2] for c in candidates]
    seq_embs, seq_masks, seq_idfs = get_embs_masks(seq_ids, seq_embs, special_ids=special_ids)
    
    scores = {'P': {}, 'R': {}, 'F': {}}
    for i, rst in enumerate(candidates):
        P, R, F = check_match(special_ids,
                              seq_embs, seq_masks, seq_idfs,
                              rst[0][:len(rst[2])], rst[2])
        scores['P'][i], scores['R'][i], scores['F'][i] = P, R, F
        max_f = max(x.item() for xid, x in enumerate(F) if xid != i)
        if max_f > .9:
            import ipdb; ipdb.set_trace()
    F_scores = list(dict(sorted(scores['F'].items(), key=lambda x: x[0])).values())
    
    return F_scores


def check_match(keys_embs: torch.Tensor, keys_masks: torch.BoolTensor, keys_idfs: torch.Tensor,
                query: list[int], query_emb: torch.Tensor,
                special_ids: list = [1, 2, 32000]):
    query_embs = torch.stack([query_emb for _ in keys_embs], dim=0)
    query_masks = torch.ones(query_embs.size()[:-1]).bool().to(query_embs.device)
    query_idfs = torch.stack([
        torch.tensor([float(_id not in special_ids) for _id in query[:query_emb.size(0)]])
    for _ in keys_idfs], dim=0).to(query_embs.device)
    
    P, R, F = greedy_cos_idf(keys_embs.float(), keys_masks, keys_idfs,
                             query_embs.float(), query_masks, query_idfs)
    return P, R, F
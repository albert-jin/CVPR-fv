"""Data pipeline for CVPR-FV.

Two datasets are produced:

* ``FVDataset`` — one FV instance × three internal prefixes (train uses
  all 9 consolidation prompts; eval uses one random template per prefix).
  Each row contains the two candidate answers (Yes / No) so that the
  decomposition head can compute the same triple loss as Det2Ver.

* ``CVPDataset`` — the CVP auxiliary corpus, built from rumor-detection
  data with the pseudo-verifiability labels produced by
  ``cvp_pseudo_labeler``. Each row is a single (claim, Yes/No) pair.

Both are unified into ``CVPRDataModule``. At training time the FV rows
and the CVP rows are interleaved in one loader with a ``task_flag``
field. At evaluation time only FV rows are yielded and grouped by
``instance_idx`` so ``model.py`` can compute the three-way heuristic score
aggregation per FV instance.
"""

import json
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tqdm import tqdm

import configs
from configs import (
    ConsoPrompts, IntPres, MapTab, DLabel2Idx, VLabelDiv, VLabelDiv_list,
    VLabel2Idx, LabelUnion, CVPPrompt, CVPLabel2Idx, CVPAnswer2Idx,
    train_file_path, test_file_path, dataset_names,
    rd_train_file_path, rd_dataset_names, cvp_cache_file_path,
    FS_CacheUse, cache_file_path,
    TOKENIZER,
    SHOT, SHOT_NUM,
)
from cvp_pseudo_labeler import label_rd_corpus


def _pad_id() -> int:
    return configs.PAD_TOKEN_ID


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def read_jsonl(file: str) -> List[dict]:
    print('read from', file)
    with open(file, encoding='utf-8', mode='rt') as inp:
        data = [json.loads(line.strip()) for line in inp if line.strip()]
    if data:
        print('data @ first row:', data[0])
    print('length:', len(data))
    return data


def save_jsonl(data: List[dict], file: str):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    print(len(data), 'rows data', 'save to', file)
    with open(file, encoding='utf-8', mode='wt') as oup:
        oup.writelines([json.dumps(d, ensure_ascii=False) + '\n' for d in data])
    print('saved.')


def label_unify(data: List[dict]) -> List[dict]:
    for d in data:
        raw = d['label']
        if raw not in VLabelDiv_list:
            d['label'] = LabelUnion.get(raw, raw)
    return data


# ---------------------------------------------------------------------------
# Fact-verification reader (identical semantics to Det2Ver)
# ---------------------------------------------------------------------------

class FVDataReader:
    def __init__(self, dataset_name: str, few_shot: bool, shot_num: int,
                 seed: int = 0, zero_shot: bool = False):
        train_path = train_file_path.format(dataset_name=dataset_name)
        test_path = test_file_path.format(dataset_name=dataset_name)
        if zero_shot:
            self.train_data = []
        elif few_shot:
            if FS_CacheUse:
                cache_file = cache_file_path(dataset_name, shot_num)
                if os.path.exists(cache_file):
                    self.train_data = label_unify(read_jsonl(cache_file))
                else:
                    self.train_data = self._sample_few_shot(train_path, shot_num, seed)
                    save_jsonl(self.train_data, cache_file)
            else:
                self.train_data = self._sample_few_shot(train_path, shot_num, seed)
        else:
            self.train_data = label_unify(read_jsonl(train_path))
        self.test_data = label_unify(read_jsonl(test_path))

    @staticmethod
    def _sample_few_shot(train_path: str, shot_num: int, seed: int) -> List[dict]:
        all_train_data = read_jsonl(train_path)
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            np.random.shuffle(all_train_data)
            support = [d for d in all_train_data if d['label'] in VLabelDiv['SUPPORT']]
            nei = [d for d in all_train_data if d['label'] in VLabelDiv['NEI']]
            refute = [d for d in all_train_data if d['label'] in VLabelDiv['REFUTE']]
            label_unify(support)
            label_unify(nei)
            label_unify(refute)
            train_data = support[:shot_num] + nei[:shot_num] + refute[:shot_num]
        finally:
            np.random.set_state(state)
        return train_data


# ---------------------------------------------------------------------------
# CVP reader — turns rumor-detection corpora into a Verifiable / Unverifiable
# corpus via ``cvp_pseudo_labeler``. Optional LLM flag callable can be
# injected from the caller.
# ---------------------------------------------------------------------------

class CVPReader:
    """Load and pseudo-label rumor detection instances for CVP."""

    def __init__(self, rd_names: List[str], total_per_dataset: int,
                 seed: int = 0, llm_flag_fn=None):
        self.rows: List[dict] = []
        per_class = max(total_per_dataset // 2, 1)
        for rd_name in rd_names:
            train_path = rd_train_file_path.format(rd_name=rd_name)
            if not os.path.exists(train_path):
                print(f'[CVPReader] {train_path} does not exist; skipping.')
                continue
            cache_file = cvp_cache_file_path(rd_name, per_class)
            if FS_CacheUse and os.path.exists(cache_file):
                cached = read_jsonl(cache_file)
                if cached:
                    print(f'[CVPReader] loaded CVP cache {cache_file} ({len(cached)} rows)')
                    self.rows.extend(cached)
                    continue

            raw_rows = read_jsonl(train_path)
            state = np.random.get_state()
            np.random.seed(seed)
            try:
                np.random.shuffle(raw_rows)
            finally:
                np.random.set_state(state)

            # Pseudo-label EVERY row first, then keep a 1:1 split.
            labelled = label_rd_corpus(raw_rows, llm_flag_fn=llm_flag_fn)
            ver = [r for r in labelled if r['cvp_label'] == 'Verifiable']
            unv = [r for r in labelled if r['cvp_label'] == 'Unverifiable']
            subset = ver[:per_class] + unv[:per_class]
            if not subset:
                print(f'[CVPReader] {rd_name}: no pseudo-labels produced (check τ / cues).')
                continue
            print(f'[CVPReader] {rd_name}: kept {len(subset)} rows '
                  f'(V={min(per_class, len(ver))} U={min(per_class, len(unv))})')
            self.rows.extend(subset)
            if FS_CacheUse:
                save_jsonl(subset, cache_file)


# ---------------------------------------------------------------------------
# Consolidation prompting (identical to Det2Ver but with additional metadata)
# ---------------------------------------------------------------------------

def consolidation_prompting(raw_instance: dict, train_flag: bool = True):
    """Expand one FV instance to |IntPres| * |ConsoPrompts| (train) or
    |IntPres| (eval) prompted variants.
    """
    variants = []
    candidate_answers = MapTab[raw_instance['label']]
    for prefix_idx, (answer, prefix) in enumerate(zip(candidate_answers, IntPres)):
        internal = prefix + raw_instance['claim']
        template_iter = enumerate(ConsoPrompts) if train_flag else (
            [(int(np.random.choice(len(ConsoPrompts))), None)]
        )
        for t_idx, _ in template_iter:
            conso = ConsoPrompts[t_idx]
            d_input = (conso.replace('[PREMISE]', raw_instance['gold_evidence_text'])
                            .replace('[New_HYPO]', internal))
            variants.append({
                'd_input': d_input,
                'd_label': answer,
                'd_label_idx': DLabel2Idx[answer],
                'v_label': raw_instance['label'],
                'v_label_idx': VLabel2Idx[raw_instance['label']],
                'int_prefix_idx': prefix_idx,
                'template_idx': t_idx,
            })
    return variants


# ---------------------------------------------------------------------------
# Torch datasets
# ---------------------------------------------------------------------------

class _AnswerCache:
    """Tokenizes the two Yes/No candidate answers once and reuses them."""

    def __init__(self, answers: List[str]):
        tok = TOKENIZER
        self.answers = answers
        self.ids = [
            tok(a, return_tensors='pt', truncation=True,
                add_special_tokens=True).input_ids.squeeze(0)
            for a in answers
        ]


class FVDataset(Dataset):
    def __init__(self, data: List[dict], train_flag: bool):
        super().__init__()
        self._answers = _AnswerCache(list(DLabel2Idx.keys()))
        self.data_list: List[dict] = []
        tok = TOKENIZER

        pbar = tqdm(data)
        pbar.set_description(f'FV {"train" if train_flag else "val"}')
        for instance_idx, d in enumerate(pbar):
            for v_idx, var in enumerate(consolidation_prompting(d, train_flag)):
                ids = tok(var['d_input'], return_tensors='pt', truncation=True,
                          add_special_tokens=True).input_ids.squeeze(0)
                self.data_list.append({
                    'input_ids': ids,
                    'd_label_idx': var['d_label_idx'],
                    'v_label_idx': var['v_label_idx'],
                    'int_prefix_idx': var['int_prefix_idx'],
                    'template_idx': var['template_idx'],
                    'instance_idx': instance_idx,
                    # Preserve the dataset identifier as well as the local row
                    # index.  The former makes cross-model analyses robust to
                    # accidental reordering of the validation JSONL file.
                    'instance_id': d.get('id', instance_idx),
                    'variant_idx': v_idx,
                    'claim': d['claim'],
                    'task_flag': 0,     # FV
                })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        row = self.data_list[idx]
        return {
            'input_ids': row['input_ids'],
            'choices_ids': self._answers.ids,
            'answer_label': row['d_label_idx'],
            'v_label_idx': row['v_label_idx'],
            'cvp_label_idx': -1,
            'int_prefix_idx': row['int_prefix_idx'],
            'template_idx': row['template_idx'],
            'instance_idx': row['instance_idx'],
            'instance_id': row['instance_id'],
            'variant_idx': row['variant_idx'],
            'claim': row['claim'],
            'task_flag': row['task_flag'],
        }


class CVPDataset(Dataset):
    """Pseudo-labelled CVP corpus. Each row = (X_cvp = CVPPrompt(c), Yes/No).

    The CVP head shares the *same backbone* as the FV head, so the row
    is emitted with the same shape as ``FVDataset`` and only differs by
    ``task_flag = 1`` and the answer vocabulary.
    """

    def __init__(self, rows: List[dict]):
        super().__init__()
        self._answers = _AnswerCache(list(CVPAnswer2Idx.keys()))
        self.data_list: List[dict] = []
        tok = TOKENIZER

        pbar = tqdm(rows)
        pbar.set_description('CVP corpus')
        for i, row in enumerate(pbar):
            cvp_label = row['cvp_label']              # Verifiable / Unverifiable
            answer_word = 'Yes.' if cvp_label == 'Verifiable' else 'No.'
            answer_idx = CVPAnswer2Idx[answer_word]
            text = CVPPrompt.replace('[CLAIM]', row.get('claim', ''))
            ids = tok(text, return_tensors='pt', truncation=True,
                      add_special_tokens=True).input_ids.squeeze(0)
            self.data_list.append({
                'input_ids': ids,
                'cvp_label_idx': CVPLabel2Idx[cvp_label],
                'answer_label': answer_idx,
                'instance_idx': -(i + 1),
                'instance_id': row.get('id', -(i + 1)),
                'variant_idx': 0,
                'claim': row.get('claim', ''),
                'task_flag': 1,
            })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        row = self.data_list[idx]
        return {
            'input_ids': row['input_ids'],
            'choices_ids': self._answers.ids,
            'answer_label': row['answer_label'],
            'v_label_idx': -1,
            'cvp_label_idx': row['cvp_label_idx'],
            'int_prefix_idx': -1,
            'template_idx': -1,
            'instance_idx': row['instance_idx'],
            'instance_id': row['instance_id'],
            'variant_idx': 0,
            'claim': row['claim'],
            'task_flag': row['task_flag'],
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_fn(batch):
    pad_id = _pad_id()
    input_ids = pad_sequence([b['input_ids'] for b in batch],
                             batch_first=True, padding_value=pad_id)
    n_choices = len(batch[0]['choices_ids'])
    padded_choices = []
    for c_idx in range(n_choices):
        choice_c = [b['choices_ids'][c_idx] for b in batch]
        padded_choices.append(pad_sequence(choice_c, batch_first=True, padding_value=pad_id))
    max_len = max(c.size(1) for c in padded_choices)
    padded_choices = [
        torch.nn.functional.pad(c, (0, max_len - c.size(1)), value=pad_id)
        for c in padded_choices
    ]
    choices_ids = torch.stack(padded_choices, dim=1)

    return {
        'input_ids': input_ids,
        'choices_ids': choices_ids,
        'answer_label': torch.LongTensor([b['answer_label'] for b in batch]),
        'v_label_idx': torch.LongTensor([b['v_label_idx'] for b in batch]),
        'cvp_label_idx': torch.LongTensor([b['cvp_label_idx'] for b in batch]),
        'task_flag': torch.LongTensor([b['task_flag'] for b in batch]),
        'meta': [{
            'int_prefix_idx': b['int_prefix_idx'],
            'template_idx': b['template_idx'],
            'instance_idx': b['instance_idx'],
            'instance_id': b['instance_id'],
            'variant_idx': b['variant_idx'],
            'claim': b['claim'],
        } for b in batch],
    }


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------

class _ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = [d for d in datasets if len(d) > 0]
        self.offsets = [0]
        for d in self.datasets:
            self.offsets.append(self.offsets[-1] + len(d))

    def __len__(self):
        return self.offsets[-1] if self.offsets else 0

    def __getitem__(self, idx):
        for i, d in enumerate(self.datasets):
            if idx < self.offsets[i + 1]:
                return d[idx - self.offsets[i]]
        raise IndexError(idx)


class CVPRDataModule(LightningDataModule):
    def __init__(self,
                 dataset_name: str = 'fever',
                 few_shot: bool = SHOT,
                 shot_num: int = SHOT_NUM,
                 seed: int = 0,
                 zero_shot: bool = False,
                 use_cvp: bool = True,
                 cvp_total_per_dataset: int = 200,
                 llm_flag_fn=None):
        super().__init__()
        assert dataset_name in dataset_names

        fv_reader = FVDataReader(dataset_name, few_shot=few_shot, shot_num=shot_num,
                                  seed=seed, zero_shot=zero_shot)
        self.fv_train_dataset = FVDataset(fv_reader.train_data, train_flag=True)
        self.test_dataset = FVDataset(fv_reader.test_data, train_flag=False)

        self.cvp_train_dataset: Optional[CVPDataset] = None
        if use_cvp and cvp_total_per_dataset > 0:
            rd_names = list(configs.rd_dataset_names_used)
            rd_only = os.environ.get('CVPRFV_RD_ONLY', '').lower().strip()
            if rd_only in configs.rd_dataset_names:
                rd_names = [rd_only]
            reader = CVPReader(rd_names, cvp_total_per_dataset, seed=seed,
                               llm_flag_fn=llm_flag_fn)
            if reader.rows:
                self.cvp_train_dataset = CVPDataset(reader.rows)

        if self.cvp_train_dataset is None:
            self.train_dataset = self.fv_train_dataset
        else:
            self.train_dataset = _ConcatDataset([self.fv_train_dataset, self.cvp_train_dataset])

        print('train:', len(self.train_dataset),
              'fv:', len(self.fv_train_dataset),
              'cvp:', 0 if self.cvp_train_dataset is None else len(self.cvp_train_dataset),
              'test:', len(self.test_dataset))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=configs.train_batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=configs.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=configs.eval_batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=configs.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()

    def cvp_inference_batch(self, claims: List[str]):
        """Instance-method alias kept for backwards compatibility."""
        return build_cvp_inference_batch(claims)


def build_cvp_inference_batch(claims: List[str]):
    """Return a collated batch that runs the CVP head on ``claims``.

    Module-level so callers do not need a ``CVPRDataModule`` instance.
    """
    tok = TOKENIZER
    answers = _AnswerCache(list(CVPAnswer2Idx.keys()))
    rows = []
    for i, claim in enumerate(claims):
        text = CVPPrompt.replace('[CLAIM]', claim)
        ids = tok(text, return_tensors='pt', truncation=True,
                  add_special_tokens=True).input_ids.squeeze(0)
        rows.append({
            'input_ids': ids,
            'choices_ids': answers.ids,
            'answer_label': 0,
            'v_label_idx': -1,
            'cvp_label_idx': 0,
            'int_prefix_idx': -1,
            'template_idx': -1,
            'instance_idx': i,
            'instance_id': i,
            'variant_idx': 0,
            'claim': claim,
            'task_flag': 1,
        })
    return collate_fn(rows)

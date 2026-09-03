"""Dataset loading, prompting-consolidation and PyTorch-Lightning
DataModule for Det2Ver.

The module exposes:

* ``V2DDataReader``: reads and, when requested, few-shot samples a
  fact-verification (FV) dataset (fever / scifact / vc).
* ``RumorDataReader``: reads and few-shot samples a rumor-detection (RD)
  dataset (liar / fnn / covid).
* ``consolidation_prompting``: implements the *Prompting Consolidation
  Mechanism* from Section III-A + Appendix A (Table VII).
* ``V2DDataSet``: expands each FV instance into ``len(IntPres) * len(ConsoPrompts)``
  training variants (or ``len(IntPres)`` random variants at eval time) and
  materialises the tokenized tensors.
* ``RumorDataSet``: expands each RD instance into two candidate answers
  (Yes / No) exactly like ``V2DDataSet`` but the input is kept unchanged
  as required by Section III-A ("we maintain the original input of rumor
  detection unchanged").
* ``V2DDataModule``: LightningDataModule wiring everything together.

Every batch produced by the data module has the shape

    (input_ids, choice_ids, d_label_idx, v_label_idx, task_flag, meta)

* ``d_label_idx``  — 0/1 gold binary label (Yes / No)
* ``v_label_idx``  — 0/1/2 ternary FV label (or -1 for RD-only rows)
* ``task_flag``    — 0 for FV, 1 for RD
* ``meta``         — per-example dict used by the label synchronisation
                    engine at evaluation time (instance id, internal
                    prefix index, template index).
"""

import json
import os
import random
from typing import List

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
    VLabel2Idx, LabelUnion, n_ways,
    train_file_path, test_file_path, dataset_names,
    rd_train_file_path, rd_dataset_names, rd_cache_file_path,
    FS_CacheUse, cache_file_path,
    TOKENIZER,
    train_batch_size, eval_batch_size, num_workers,
    SHOT, SHOT_NUM,
)


def _pad_id() -> int:
    """Look up the current pad token id at call time so that a later
    tokenizer load (which updates ``configs.PAD_TOKEN_ID``) is reflected."""
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
# Fact-Verification data reader (few-shot / zero-shot aware)
# ---------------------------------------------------------------------------

class V2DDataReader:
    def __init__(self, dataset_name: str, few_shot: bool, shot_num: int, seed: int = 0,
                 zero_shot: bool = False):
        """Read one of the three FV datasets and, when in few-shot mode,
        cache-persist the sampled subset so that all experiments with the
        same ``(dataset, shot_num, seed)`` triple see the same examples.

        :param dataset_name: 'fever' / 'scifact' / 'vc'
        :param few_shot:     whether to use only ``shot_num`` examples per class
        :param shot_num:     K in K-shot (per class)
        :param seed:         random seed controlling the sample
        :param zero_shot:    if True the training set is emptied
        """
        train_path = train_file_path.format(dataset_name=dataset_name)
        test_path = test_file_path.format(dataset_name=dataset_name)

        if zero_shot:
            self.train_data = []
        elif few_shot:
            if FS_CacheUse:
                cache_file = cache_file_path(dataset_name, shot_num)
                if os.path.exists(cache_file):
                    print(f'load from cache {cache_file}...')
                    self.train_data = label_unify(read_jsonl(cache_file))
                    print('loaded.')
                else:
                    print('cache not exist, random select from train set.')
                    self.train_data = self._sample_few_shot(train_path, shot_num, seed)
                    print('creating cache...')
                    save_jsonl(self.train_data, cache_file)
                    print('cache created.')
            else:
                self.train_data = self._sample_few_shot(train_path, shot_num, seed)
        else:
            self.train_data = label_unify(read_jsonl(train_path))

        self.test_data = label_unify(read_jsonl(test_path))

    @staticmethod
    def _sample_few_shot(train_path: str, shot_num: int, seed: int) -> List[dict]:
        all_train_data = read_jsonl(train_path)
        # Deterministic sampling controlled by ``seed``.
        state = np.random.get_state()
        np.random.seed(seed)
        np.random.shuffle(all_train_data)
        try:
            support = [d for d in all_train_data if d['label'] in VLabelDiv['SUPPORT']]
            nei = [d for d in all_train_data if d['label'] in VLabelDiv['NEI']]
            refute = [d for d in all_train_data if d['label'] in VLabelDiv['REFUTE']]
            label_unify(support)
            label_unify(nei)
            label_unify(refute)
            train_data = support[:shot_num] + nei[:shot_num] + refute[:shot_num]
        finally:
            np.random.set_state(state)
        assert len(train_data) == shot_num * n_ways, (
            f'few-shot sampling produced {len(train_data)} rows, expected {shot_num * n_ways}'
        )
        return train_data


# ---------------------------------------------------------------------------
# Rumor-Detection data reader
# ---------------------------------------------------------------------------

class RumorDataReader:
    """Loader for one of the three rumor-detection corpora.

    A row must have ``{"id", "claim", "label"}``.
    ``label`` must be one of ``{"FAKE", "REAL"}`` (case-insensitive).
    """

    LABEL_TO_ANSWER = {'REAL': 'Yes, it is.', 'FAKE': "No, it isn't."}

    def __init__(self, rd_name: str, num_per_class: int, seed: int = 0):
        assert rd_name in rd_dataset_names, (
            f'rd_name must be one of {rd_dataset_names}, got {rd_name}'
        )
        self.rd_name = rd_name
        self.num_per_class = num_per_class

        train_path = rd_train_file_path.format(rd_name=rd_name)
        if not os.path.exists(train_path):
            print(f'[RumorDataReader] {train_path} does not exist; falling back to empty split. '
                  f'See data/rumor/README.md for download instructions.')
            self.train_data = []
            return

        cache_file = rd_cache_file_path(rd_name, num_per_class)
        if FS_CacheUse and os.path.exists(cache_file):
            print(f'load rumor cache {cache_file}...')
            self.train_data = self._normalize(read_jsonl(cache_file))
        else:
            all_rd = self._normalize(read_jsonl(train_path))
            state = np.random.get_state()
            np.random.seed(seed)
            np.random.shuffle(all_rd)
            try:
                real = [d for d in all_rd if d['label'] == 'REAL']
                fake = [d for d in all_rd if d['label'] == 'FAKE']
                self.train_data = real[:num_per_class] + fake[:num_per_class]
            finally:
                np.random.set_state(state)
            if FS_CacheUse:
                save_jsonl(self.train_data, cache_file)

    @staticmethod
    def _normalize(data: List[dict]) -> List[dict]:
        out = []
        for d in data:
            raw = str(d.get('label', '')).strip().upper()
            if raw in ('REAL', 'TRUE', '1', 'NOT FAKE', 'RELIABLE'):
                d['label'] = 'REAL'
            elif raw in ('FAKE', 'FALSE', '0', 'PANTS-FIRE', 'PANTS ON FIRE',
                         'MOSTLY-FALSE', 'BARELY-TRUE', 'UNRELIABLE'):
                d['label'] = 'FAKE'
            else:
                # 'half-true', 'mostly-true' etc. — skip
                continue
            if 'claim' not in d:
                d['claim'] = d.get('text') or d.get('statement') or ''
            out.append(d)
        return out


# ---------------------------------------------------------------------------
# Prompting Consolidation Mechanism (Section III-A, Appendix A/B)
# ---------------------------------------------------------------------------

def consolidation_prompting(raw_instance: dict, train_flag: bool = True) -> List[dict]:
    """Expand one FV instance into ``|IntPres| * |ConsoPrompts|`` (train)
    or ``|IntPres|`` (eval) prompted variants.

    Each variant carries the intermediate answer expected by the label
    mapping table so that ``V2DDataSet`` can feed the model a
    (input, target) pair.
    """
    variant_instances = []
    candidate_answers = MapTab[raw_instance['label']]
    instance_temp = [
        {'internal_d_input': prefix + raw_instance['claim'],
         'd_label': answer,
         'v_label': raw_instance['label'],
         'int_prefix_idx': idx}
        for idx, (answer, prefix) in enumerate(zip(candidate_answers, IntPres))
    ]
    if train_flag:
        for instance in instance_temp:
            for t_idx, conso_prompt in enumerate(ConsoPrompts):
                d_input = (conso_prompt
                           .replace('[PREMISE]', raw_instance['gold_evidence_text'])
                           .replace('[New_HYPO]', instance['internal_d_input']))
                variant_instances.append({
                    'd_input': d_input,
                    'd_label': instance['d_label'],
                    'd_label_idx': DLabel2Idx[instance['d_label']],
                    'v_label': instance['v_label'],
                    'v_label_idx': VLabel2Idx[instance['v_label']],
                    'int_prefix_idx': instance['int_prefix_idx'],
                    'template_idx': t_idx,
                })
    else:
        # Use one random consolidation prompt per internal prefix at eval.
        for instance in instance_temp:
            t_idx = int(np.random.choice(len(ConsoPrompts)))
            conso_prompt = ConsoPrompts[t_idx]
            d_input = (conso_prompt
                       .replace('[PREMISE]', raw_instance['gold_evidence_text'])
                       .replace('[New_HYPO]', instance['internal_d_input']))
            variant_instances.append({
                'd_input': d_input,
                'd_label': instance['d_label'],
                'd_label_idx': DLabel2Idx[instance['d_label']],
                'v_label': instance['v_label'],
                'v_label_idx': VLabel2Idx[instance['v_label']],
                'int_prefix_idx': instance['int_prefix_idx'],
                'template_idx': t_idx,
            })
    return variant_instances


# ---------------------------------------------------------------------------
# Torch Datasets
# ---------------------------------------------------------------------------

class V2DDataSet(Dataset):
    """Expanded FV dataset producing (input, choices, labels) rows.

    Each row exposes both possible target answers so that the model can
    compute the unlikelihood and length-normalization losses jointly.
    """

    def __init__(self, data: List[dict], train_flag: bool):
        super().__init__()
        self.tokenizer = TOKENIZER
        self.train_flag = train_flag
        self.data_list: List[dict] = []
        self._all_answers = list(DLabel2Idx.keys())  # ['Yes, it is.', "No, it isn't."]
        self._all_answer_ids = [
            self.tokenizer(a, return_tensors='pt', truncation=True,
                           add_special_tokens=True).input_ids.squeeze(0)
            for a in self._all_answers
        ]

        pbar = tqdm(data)
        pbar.set_description(f'Processing {"train" if train_flag else "validation"} FV set')
        for instance_idx, d in enumerate(pbar):
            variants = consolidation_prompting(d, train_flag=train_flag)
            for v_idx, var_inst in enumerate(variants):
                d_input_ids = self.tokenizer(
                    var_inst['d_input'], return_tensors='pt', truncation=True,
                    add_special_tokens=True,
                ).input_ids.squeeze(0)
                self.data_list.append({
                    'd_input_ids': d_input_ids,
                    'd_label_idx': var_inst['d_label_idx'],
                    'v_label_idx': var_inst['v_label_idx'],
                    'int_prefix_idx': var_inst['int_prefix_idx'],
                    'template_idx': var_inst['template_idx'],
                    'instance_idx': instance_idx,
                    # Keep the source dataset ID for stable joins with other
                    # models; ``instance_idx`` remains for within-run grouping.
                    'instance_id': d.get('id', instance_idx),
                    'variant_idx': v_idx,
                    'claim': d.get('claim', ''),
                    'task_flag': 0,  # FV
                })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        row = self.data_list[idx]
        return {
            'd_input_ids': row['d_input_ids'],
            'choices_ids': self._all_answer_ids,
            'd_label_idx': row['d_label_idx'],
            'v_label_idx': row['v_label_idx'],
            'int_prefix_idx': row['int_prefix_idx'],
            'template_idx': row['template_idx'],
            'instance_idx': row['instance_idx'],
            'instance_id': row['instance_id'],
            'variant_idx': row['variant_idx'],
            'claim': row['claim'],
            'task_flag': row['task_flag'],
        }


class RumorDataSet(Dataset):
    """Rumor-detection dataset. The input is kept **unchanged** (Section III-A).

    Each rumor instance is exposed as a single row with the
    Yes/No candidate answers, i.e. the model can be trained with the same
    ``LM + Unlikelihood + Length-Norm`` triple loss.
    """

    def __init__(self, data: List[dict]):
        super().__init__()
        self.tokenizer = TOKENIZER
        self.data_list: List[dict] = []
        self._all_answers = list(DLabel2Idx.keys())
        self._all_answer_ids = [
            self.tokenizer(a, return_tensors='pt', truncation=True,
                           add_special_tokens=True).input_ids.squeeze(0)
            for a in self._all_answers
        ]

        pbar = tqdm(data)
        pbar.set_description('Processing rumor detection set')
        for i, d in enumerate(pbar):
            answer_word = RumorDataReader.LABEL_TO_ANSWER[d['label']]
            answer_idx = DLabel2Idx[answer_word]
            d_input_ids = self.tokenizer(
                d['claim'], return_tensors='pt', truncation=True,
                add_special_tokens=True,
            ).input_ids.squeeze(0)
            self.data_list.append({
                'd_input_ids': d_input_ids,
                'd_label_idx': answer_idx,
                'v_label_idx': -1,          # RD has no ternary label
                'int_prefix_idx': -1,
                'template_idx': -1,
                'instance_idx': -(i + 1),   # negative to distinguish from FV
                'instance_id': d.get('id', -(i + 1)),
                'variant_idx': 0,
                'claim': d.get('claim', ''),
                'task_flag': 1,             # RD
            })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        row = self.data_list[idx]
        return {
            'd_input_ids': row['d_input_ids'],
            'choices_ids': self._all_answer_ids,
            'd_label_idx': row['d_label_idx'],
            'v_label_idx': row['v_label_idx'],
            'int_prefix_idx': row['int_prefix_idx'],
            'template_idx': row['template_idx'],
            'instance_idx': row['instance_idx'],
            'instance_id': row['instance_id'],
            'variant_idx': row['variant_idx'],
            'claim': row['claim'],
            'task_flag': row['task_flag'],
        }


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

def collate_fn(batch):
    pad_id = _pad_id()
    d_input_ids_batch = pad_sequence([b['d_input_ids'] for b in batch],
                                     batch_first=True, padding_value=pad_id)
    n_choices = len(batch[0]['choices_ids'])
    padded_choices = []
    for c_idx in range(n_choices):
        choice_c = [b['choices_ids'][c_idx] for b in batch]
        padded_choices.append(pad_sequence(choice_c, batch_first=True, padding_value=pad_id))
    # (B, n_choices, L_choice)
    max_len = max(c.size(1) for c in padded_choices)
    padded_choices = [
        torch.nn.functional.pad(c, (0, max_len - c.size(1)), value=pad_id)
        for c in padded_choices
    ]
    choices_ids_batch = torch.stack(padded_choices, dim=1)

    d_label_idx = torch.LongTensor([b['d_label_idx'] for b in batch])
    v_label_idx = torch.LongTensor([b['v_label_idx'] for b in batch])
    task_flag = torch.LongTensor([b['task_flag'] for b in batch])
    meta = [{
        'int_prefix_idx': b['int_prefix_idx'],
        'template_idx': b['template_idx'],
        'instance_idx': b['instance_idx'],
        'instance_id': b['instance_id'],
        'variant_idx': b['variant_idx'],
        'claim': b['claim'],
    } for b in batch]

    return {
        'input_ids': d_input_ids_batch,
        'choices_ids': choices_ids_batch,
        'd_label_idx': d_label_idx,
        'v_label_idx': v_label_idx,
        'task_flag': task_flag,
        'meta': meta,
    }


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------

class ConcatWithIdxDataset(Dataset):
    """Plain ``torch.utils.data.ConcatDataset`` alternative that keeps
    tensor rows dict-shaped."""

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


class V2DDataModule(LightningDataModule):
    """LightningDataModule combining fact-verification and rumor-detection
    training corpora. The validation corpus is FV-only."""

    def __init__(self,
                 dataset_name: str = 'fever',
                 few_shot: bool = SHOT,
                 shot_num: int = SHOT_NUM,
                 seed: int = 0,
                 zero_shot: bool = False,
                 use_rumor_detection: bool = True,
                 rd_total_per_dataset: int = 20):
        super().__init__()
        assert dataset_name in dataset_names

        datareader = V2DDataReader(
            dataset_name, few_shot=few_shot, shot_num=shot_num,
            seed=seed, zero_shot=zero_shot,
        )
        self.train_data = datareader.train_data
        self.test_data = datareader.test_data

        self.fv_train_dataset = V2DDataSet(self.train_data, train_flag=True)
        self.test_dataset = V2DDataSet(self.test_data, train_flag=False)

        rd_datasets = []
        if use_rumor_detection and rd_total_per_dataset > 0:
            per_class = max(rd_total_per_dataset // 2, 1)
            # Single-source ablation (Figure 3): if ``DET2VER_RD_ONLY`` is set,
            # only that RD dataset is used; the K per class is unchanged so
            # the total number of RD instances matches ``rd_total_per_dataset``.
            rd_only = os.environ.get('DET2VER_RD_ONLY', '').lower().strip()
            active_rd = [rd_only] if rd_only in rd_dataset_names else rd_dataset_names
            for rd_name in active_rd:
                rr = RumorDataReader(rd_name, per_class, seed=seed)
                if rr.train_data:
                    rd_datasets.append(RumorDataSet(rr.train_data))

        self.rd_train_dataset = ConcatWithIdxDataset(rd_datasets) if rd_datasets else None
        if self.rd_train_dataset is None:
            self.train_dataset = self.fv_train_dataset
        else:
            self.train_dataset = ConcatWithIdxDataset([self.fv_train_dataset, self.rd_train_dataset])

        print('train_dataset:', len(self.train_dataset),
              'fv:', len(self.fv_train_dataset),
              'rd:', 0 if self.rd_train_dataset is None else len(self.rd_train_dataset),
              'test_dataset:', len(self.test_dataset))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, batch_size=train_batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset, batch_size=eval_batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, drop_last=False,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()


# ---------------------------------------------------------------------------
# quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    fever = V2DDataModule(dataset_name='fever', few_shot=True, shot_num=4,
                           use_rumor_detection=True, rd_total_per_dataset=20)
    for batch in fever.train_dataloader():
        print('train batch:', {k: v if not isinstance(v, torch.Tensor) else v.shape
                                for k, v in batch.items()})
        break
    for batch in fever.val_dataloader():
        print('val batch:', {k: v if not isinstance(v, torch.Tensor) else v.shape
                              for k, v in batch.items()})
        break

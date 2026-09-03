"""Global configuration for Det2Ver.

This file collects every tunable knob used by the framework: prompt
templates, label spaces, PLM checkpoints, LoRA / (IA)^3 hyperparameters,
optimizer schedule and the auxiliary rumor-detection datasets used for
cross-task knowledge transfer.

Command-line arguments in ``train.py`` override the module-level values
below via ``configs.<name> = value`` before any downstream import that
relies on the value.
"""

import os
import socket

# ---------------------------------------------------------------------------
# 1. Prompting Consolidation Mechanism (Appendix A, Table VII)
# ---------------------------------------------------------------------------

# The nine consolidation prompts used to lift a (premise, claim) pair into
# a single sequence that looks like a rumor-detection input.
ConsoPrompts = [
    "Taking the [PREMISE] into account, is it possible to ascertain the truth of the [New_HYPO]?",
    "[PREMISE] Using only the above PREMISE and what you know, can we induced that [New_HYPO] is true?",
    "Given that [PREMISE] \n Therefore, it must be true that [New_HYPO]? Yes or No?",
    "Suppose [PREMISE], Can we infer that [New_HYPO]? Yes or No?",
    "Considering the [PREMISE] and applying our knowledge, is it reasonable that [New_HYPO] is accurate?",
    "[PREMISE] Keeping in mind the above text, consider: [New_HYPO] \n Is this true? Yes or No?",
    "In light of the [PREMISE], can we establish that the [New_HYPO] is indeed factual? Yes or No?",
    "[PREMISE] Based on that information, can we conject that the claim: [New_HYPO] is established?",
    "Based on the [PREMISE] provided and our understanding, can we infer that the [New_HYPO] holds true?",
]

# Internal Prefixes injected between the consolidation template and the
# hypothesis / claim. They decompose the ternary verification label into a
# collection of binary yes/no sub-decisions.
IntPres = ['It is true that ', 'It is uncertain that ', 'It is false that ']

# ---------------------------------------------------------------------------
# 2. Label Words Synchronization Engine (Table I, Section III-C)
# ---------------------------------------------------------------------------

# Row = fact-verification label, columns = internal prefix (true / uncertain / false).
MapTab = {
    'SUPPORT': ['Yes, it is.', "No, it isn't.", "No, it isn't."],
    'NEI':     ["No, it isn't.", 'Yes, it is.', "No, it isn't."],
    'REFUTE':  ["No, it isn't.", "No, it isn't.", 'Yes, it is.'],
}
DLabel2Idx = {'Yes, it is.': 0, "No, it isn't.": 1}
Idx2DLabel = {v: k for k, v in DLabel2Idx.items()}

# Fact Verification Labels
Labels = list(MapTab.keys())
n_ways = len(Labels)  # three classes for FV

# Original Labels unified to the pre-defined labels used in MapTab.
LabelUnion = {
    'SUPPORTS': 'SUPPORT', 'SUPPORTED': 'SUPPORT', 'SUPPORT': 'SUPPORT',
    'REFUTED': 'REFUTE', 'REFUTES': 'REFUTE', 'CONTRADICT': 'REFUTE',
    'REFUTE': 'REFUTE', 'NEI': 'NEI', 'NOT ENOUGH INFO': 'NEI',
}

VLabelDiv = {
    'SUPPORT': ['SUPPORTS', 'SUPPORTED', 'SUPPORT'],
    'REFUTE':  ['CONTRADICT', 'REFUTES', 'REFUTED', 'REFUTE'],
    'NEI':     ['NEI'],
}
VLabelDiv_list = list(VLabelDiv)
VLabel2Idx = {label: VLabelDiv_list.index(label) for label in VLabelDiv_list}
Idx2VLabel = {v: k for k, v in VLabel2Idx.items()}

# ---------------------------------------------------------------------------
# 3. Data locations
# ---------------------------------------------------------------------------

_this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.environ.get(
    'DET2VER_DATA_DIR',
    os.path.abspath(os.path.join(_this_dir, os.pardir, 'data')),
)

# Fact-verification (Section IV-B, Table II).
train_file_path = os.path.join(data_dir, '{dataset_name}_train.jsonl')
test_file_path = os.path.join(data_dir, '{dataset_name}_validation.jsonl')
dataset_names = ['fever', 'scifact', 'vc']

# Rumor-detection (Section IV-B): LIAR, FakeNewsNet, COVID-19 Fake News.
rd_train_file_path = os.path.join(data_dir, 'rumor', '{rd_name}_train.jsonl')
rd_test_file_path = os.path.join(data_dir, 'rumor', '{rd_name}_validation.jsonl')
rd_dataset_names = ['liar', 'fnn', 'covid']

# Where to write Det2Ver-shaped merged rumor detection cache.
rd_cache_file_path = lambda rd_name, num_per_class: os.path.join(
    data_dir, 'rumor', 'few_shot', rd_name, f'{num_per_class}-per-class.jsonl'
)

# Few-shot cache for fact verification. Same layout as ProToCo.
FS_CacheUse = True
cache_file_path = lambda dataset_name, few_shot_num: os.path.join(
    data_dir, 'few_shot', dataset_name, f'{few_shot_num}-shot.jsonl'
)

# ---------------------------------------------------------------------------
# 4. PLM backbone
# ---------------------------------------------------------------------------

# Local T0-3B checkpoint (any HuggingFace snapshot with the ``bigscience/T0_3B``
# layout works, e.g. cloned from https://huggingface.co/bigscience/T0_3B).
pretrained_model_path_linux = os.environ.get(
    'DET2VER_T0_PATH_LINUX', '/home/jinwq/PLMs/bigscience/T0_3B'
)
pretrained_model_path_win = os.environ.get(
    'DET2VER_T0_PATH_WIN', r'C:\Users\Weiqiang Jin\Desktop\fake detection\PLMs\bigscience\T0_3B'
)
_is_win = socket.gethostname() == "DESKTOP-ANP1MI8" or os.name == 'nt'
pretrained_model_path = pretrained_model_path_win if _is_win else pretrained_model_path_linux

# Fall back to the HuggingFace hub identifier when neither local path exists.
if not os.path.isdir(pretrained_model_path):
    pretrained_model_path = 'bigscience/T0_3B'
    _load_local_only = False
else:
    _load_local_only = True

# Tokenizer is loaded lazily by ``get_tokenizer()`` so this module can be
# imported cheaply during unit tests.
_TOKENIZER = None
PAD_TOKEN_ID = 0  # T0/T5 pad id, will be reset when the tokenizer is loaded


def get_tokenizer():
    """Return a cached HuggingFace tokenizer for the configured backbone."""
    global _TOKENIZER, PAD_TOKEN_ID
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        print('load LLM Tokenizer...')
        _TOKENIZER = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            local_files_only=_load_local_only,
        )
        _TOKENIZER.model_max_length = 250  # per paper Section IV-E
        PAD_TOKEN_ID = _TOKENIZER.pad_token_id
        print('tokenizer loaded.')
    return _TOKENIZER


# Backwards-compatible alias used by ``data_reader.py``.
class _TokenizerProxy:
    def __getattr__(self, item):
        return getattr(get_tokenizer(), item)

    def __call__(self, *args, **kwargs):
        return get_tokenizer()(*args, **kwargs)


TOKENIZER = _TokenizerProxy()

# ---------------------------------------------------------------------------
# 5. Few-shot / zero-shot regime
# ---------------------------------------------------------------------------

# By default we sample ``SHOT_NUM`` instances *per class* from the FV
# training corpus (matching Section IV-E: K in {4, 8, 16, 32}).
SHOT = True
SHOT_NUM = 4

# Number of rumor-detection instances *per RD dataset* that are folded
# into training (paper: 20, 50 or 100 total examples per dataset are the
# reported settings; Det2Ver(20) / Det2Ver(50) / Det2Ver(100)).
# ``rd_shot_num_per_class`` divides this equally between real / fake.
rd_total_per_dataset = 20
rd_num_datasets = len(rd_dataset_names)

# When True, the training corpus is fact verification only (T-Few baseline
# behaviour used to reproduce Det2Ver(0)).
use_rumor_detection = True

# Zero-shot flag: do NOT sample any FV example, rely only on the RD signal
# (Section V-B, Table IV).
zero_shot = False

# ---------------------------------------------------------------------------
# 6. Batch / trainer hyperparameters
# ---------------------------------------------------------------------------

train_batch_size = 1
eval_batch_size = 4
grad_accum_factor = 2
grad_clip_norm = 1.0
compute_precision = 'fp32'
num_workers = 0  # Windows-safe default; increase on Linux

# ---------------------------------------------------------------------------
# 7. LoRA / (IA)^3 configuration (Section III-B)
# ---------------------------------------------------------------------------

lora_modules = ".*SelfAttention|.*EncDecAttention|.*DenseReluDense"
lora_layers = "k|v|wi_1.*"
lora_rank = 0
lora_scaling_rank = 1
lora_init_scale = 0.0

# Alternative genuine LoRA setting kept for the ablations.
_lora_rank = 4
_lora_layers = "q|k|v|o|w.*"
_lora_scaling_rank = 0
_lora_init_scale = 0.01

# Trainable parameter regex (matches (IA)^3 layer B by default).
trainable_param_names_re = ".*lora_b.*"

# ---------------------------------------------------------------------------
# 8. Optimization schedule (Section IV-E)
# ---------------------------------------------------------------------------

lr = 1e-4
weight_decay = 0.0
scale_parameter = True
scheduler = 'linear_decay_with_warmup'
warmup_ratio = 0.06
num_steps = 1500
eval_step_interval = 50   # test every 50 optimizer steps
patience = 5              # early stopping: 5 evaluations without improvement

# ---------------------------------------------------------------------------
# 9. Reproducibility & experiment tracking
# ---------------------------------------------------------------------------

seed = 0
exp_root = os.path.join(_this_dir, 'output')
exp_name = 'det2ver_run'
save_model = True
load_weight = ''  # path to a checkpoint to warm-start from (e.g. t0 ia3 init)

"""Global configuration for CVPR-FV.

CVPR-FV = Claim Verifiability Prediction-based Rumor detection for Fact
Verification. See main.tex Section 3.

The framework has two learnable components:

1. **CVP (Claim Verifiability Prediction)** — auxiliary binary classifier
   whose target space is ``{Verifiable, Unverifiable}``. It is trained on
   rumor-detection corpora with pseudo-verifiability labels derived from
   five heuristic language cues plus one LLM-driven flag (weights
   ``1/1/1/1/1/2``, unverifiability threshold ``τ = 2``).

2. **Decomposition-based fact verifier** — reuses Det2Ver's three-prefix
   consolidation prompting to produce three binary confidences
   ``q_true, q_false, q_uncertain``.

Both components share a LoRA-adapted PLM (T0-3B by default; Qwen2.5-3B
and Llama-3.1-8B are also supported).

The final label is obtained via the **CVP-guided probabilistic
aggregation** (Section 3.2 of the paper), *not* via Det2Ver's hard
lookup table.
"""

import os
import socket

# ---------------------------------------------------------------------------
# 1. Fact-Verification prompting (inherited from Det2Ver)
# ---------------------------------------------------------------------------

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

# Internal prefixes = three hypothesis states s_k in {true, uncertain, false}.
IntPres = ['It is true that ', 'It is uncertain that ', 'It is false that ']

# Fact verification label vocabulary and index maps.
Labels = ['SUPPORT', 'REFUTE', 'NEI']
n_ways = len(Labels)

VLabelDiv = {
    'SUPPORT': ['SUPPORTS', 'SUPPORTED', 'SUPPORT'],
    'REFUTE':  ['CONTRADICT', 'REFUTES', 'REFUTED', 'REFUTE'],
    'NEI':     ['NEI'],
}
VLabelDiv_list = list(VLabelDiv)
VLabel2Idx = {label: VLabelDiv_list.index(label) for label in VLabelDiv_list}
Idx2VLabel = {v: k for k, v in VLabel2Idx.items()}

LabelUnion = {
    'SUPPORTS': 'SUPPORT', 'SUPPORTED': 'SUPPORT', 'SUPPORT': 'SUPPORT',
    'REFUTED': 'REFUTE', 'REFUTES': 'REFUTE', 'CONTRADICT': 'REFUTE',
    'REFUTE': 'REFUTE', 'NEI': 'NEI', 'NOT ENOUGH INFO': 'NEI',
}

# Yes/No answer vocabulary for the binary FV decomposition.
DLabel2Idx = {'Yes, it is.': 0, "No, it isn't.": 1}
Idx2DLabel = {v: k for k, v in DLabel2Idx.items()}

# Every ternary label maps back to the yes/no answer expected under each
# internal prefix (Det2Ver Table I). CVPR-FV uses these only for training the
# decomposition head; the final ternary label is computed by the
# probabilistic aggregation, not this table.
MapTab = {
    'SUPPORT': ['Yes, it is.', "No, it isn't.", "No, it isn't."],
    'NEI':     ["No, it isn't.", 'Yes, it is.', "No, it isn't."],
    'REFUTE':  ["No, it isn't.", "No, it isn't.", 'Yes, it is.'],
}

# ---------------------------------------------------------------------------
# 2. CVP (Claim Verifiability Prediction) auxiliary task
# ---------------------------------------------------------------------------

CVPLabels = ['Verifiable', 'Unverifiable']
CVPLabel2Idx = {'Verifiable': 0, 'Unverifiable': 1}
Idx2CVPLabel = {v: k for k, v in CVPLabel2Idx.items()}

# Answer surface forms for the CVP head.
CVPAnswer2Idx = {'Yes.': 0, 'No.': 1}
Idx2CVPAnswer = {v: k for k, v in CVPAnswer2Idx.items()}

# Prompt template X_cvp = [CVP-PROMPT] ⊕ c (main.tex Section 3.1).
CVPPrompt = (
    "You are given a standalone claim without any external evidence. "
    "Judge whether the claim is intrinsically verifiable — i.e., whether it "
    "makes an objective, checkable statement of fact. "
    "Answer Yes if the claim is verifiable, No otherwise.\n"
    "Claim: [CLAIM]\n"
    "Is this claim verifiable? Answer Yes or No."
)

# Heuristic cues (a)-(e) used by the pseudo-verifiability rule; each has
# weight 1 in u(c). Cue (f) is an LLM flag with weight 2.
CVP_CUES = [
    # (a) normative / prescriptive language
    ('normative',   ['should', 'ought to', 'must', 'need to', 'have to',
                     'better ', 'worse ']),
    # (b) vague quantifiers / hedges
    ('vague',       ['many', 'some ', 'few ', 'often', 'usually', 'sometimes',
                     'generally', 'commonly', 'a lot of', 'lots of',
                     'a number of', 'plenty of', 'various']),
    # (c) absolutist language
    ('absolutist',  ['always', 'never', 'everyone', 'nobody', 'nothing',
                     'everything', 'all people', 'no one', 'entire',
                     'completely', 'absolutely', 'totally']),
    # (d) causal claims without mechanism
    ('causal',      ['causes', 'because of', 'due to', 'leads to',
                     'results in', 'triggers', 'makes people']),
    # (e) unverifiable attributions
    ('attribution', ['experts believe', 'studies show', 'a study shows',
                     'research suggests', 'some say', 'it is said',
                     'many believe', 'people think', 'sources claim']),
]
CVP_CUE_WEIGHTS = [1, 1, 1, 1, 1]     # weight of cues (a)-(e)
CVP_LLM_WEIGHT = 2                    # weight of cue (f)
CVP_TAU = 2                           # unverifiability threshold τ

# ---------------------------------------------------------------------------
# 3. Probabilistic aggregation hyperparameters (main.tex Section 3.2)
# ---------------------------------------------------------------------------

lam_prior = 0.5          # λ — influence of verifiability prior
nei_floor_gamma = 0.1    # γ — small constant keeping NEI mass non-zero
lam_cvp = 1.0            # λ_cvp — weight on CVP loss in joint objective

# ---------------------------------------------------------------------------
# 4. Data locations
# ---------------------------------------------------------------------------

_this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(_this_dir, 'data')

train_file_path = os.path.join(data_dir, '{dataset_name}_train.jsonl')
test_file_path = os.path.join(data_dir, '{dataset_name}_validation.jsonl')
dataset_names = ['fever', 'scifact', 'vc']

rd_train_file_path = os.path.join(data_dir, 'rumor', '{rd_name}_train.jsonl')
rd_test_file_path = os.path.join(data_dir, 'rumor', '{rd_name}_validation.jsonl')
rd_dataset_names = ['liar', 'fnn', 'covid']

cvp_cache_file_path = lambda rd_name, num_per_class: os.path.join(
    data_dir, 'rumor', 'cvp_cache', rd_name, f'{num_per_class}-per-class.jsonl'
)
rd_cache_file_path = lambda rd_name, num_per_class: os.path.join(
    data_dir, 'rumor', 'few_shot', rd_name, f'{num_per_class}-per-class.jsonl'
)

FS_CacheUse = True
cache_file_path = lambda dataset_name, few_shot_num: os.path.join(
    data_dir, 'few_shot', dataset_name, f'{few_shot_num}-shot.jsonl'
)

# ---------------------------------------------------------------------------
# 5. PLM backbones
# ---------------------------------------------------------------------------

BACKBONES = {
    't0-3b':        'bigscience/T0_3B',
    'qwen2.5-3b':   'Qwen/Qwen2.5-3B-Instruct',
    'llama-3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
}
BACKBONE_KIND = {
    't0-3b':        'seq2seq',
    'qwen2.5-3b':   'causal',
    'llama-3.1-8b': 'causal',
}
backbone = 't0-3b'

pretrained_model_path_env = {
    't0-3b':        'CVPRFV_T0_PATH',
    'qwen2.5-3b':   'CVPRFV_QWEN_PATH',
    'llama-3.1-8b': 'CVPRFV_LLAMA_PATH',
}


def resolve_backbone_path(name: str = None) -> str:
    name = (name or backbone).lower()
    env_key = pretrained_model_path_env.get(name)
    if env_key and os.environ.get(env_key) and os.path.isdir(os.environ[env_key]):
        return os.environ[env_key]
    return BACKBONES[name]


pretrained_model_path = resolve_backbone_path()

_TOKENIZER = None
PAD_TOKEN_ID = 0


def get_tokenizer():
    global _TOKENIZER, PAD_TOKEN_ID
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        path = resolve_backbone_path()
        print(f'load tokenizer from {path} ...')
        _TOKENIZER = AutoTokenizer.from_pretrained(
            path, use_fast=True, local_files_only=os.path.isdir(path),
        )
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
        _TOKENIZER.model_max_length = 250
        PAD_TOKEN_ID = _TOKENIZER.pad_token_id
        print('tokenizer loaded.')
    return _TOKENIZER


class _TokenizerProxy:
    def __getattr__(self, item):
        return getattr(get_tokenizer(), item)

    def __call__(self, *args, **kwargs):
        return get_tokenizer()(*args, **kwargs)


TOKENIZER = _TokenizerProxy()

# ---------------------------------------------------------------------------
# 6. Few-shot / zero-shot regime
# ---------------------------------------------------------------------------

SHOT = True
SHOT_NUM = 4
zero_shot = False
use_cvp = True          # False = ablate CVP → reverts to decomposition only

# The paper reports ~200 auxiliary CVP training examples per rumor dataset,
# 1:1 verifiable/unverifiable ratio (Section 4.1 Implementation Details).
cvp_total_per_dataset = 200
rd_dataset_names_used = list(rd_dataset_names)

# ---------------------------------------------------------------------------
# 7. Batch / trainer hyperparameters (main.tex Implementation Details)
# ---------------------------------------------------------------------------

train_batch_size = 8
eval_batch_size = 8
grad_accum_factor = 1
grad_clip_norm = 1.0
compute_precision = 'bf16'
num_workers = 0

# ---------------------------------------------------------------------------
# 8. LoRA configuration
# ---------------------------------------------------------------------------

lora_modules_seq2seq = ".*SelfAttention|.*EncDecAttention|.*DenseReluDense"
lora_layers_seq2seq = "k|v|wi_1.*"
lora_modules_causal = ".*self_attn|.*mlp"
lora_layers_causal = "q_proj|k_proj|v_proj|o_proj"

lora_rank = 8
lora_scaling_rank = 0
lora_init_scale = 0.01
lora_dropout = 0.0

# Match LoRA A and B parameters (and legacy (IA)^3 multi_lora_[ab] tensors).
trainable_param_names_re = ".*lora_[ab].*|.*multi_lora_[ab].*"

# ---------------------------------------------------------------------------
# 9. Optimization schedule (main.tex Implementation Details)
# ---------------------------------------------------------------------------

lr = 1e-5
weight_decay = 0.0
scale_parameter = True
scheduler = 'linear_decay_with_warmup'
warmup_ratio = 0.06
num_epochs = 10
num_steps = 1500
eval_step_interval = 50
patience = 5

# ---------------------------------------------------------------------------
# 10. Reproducibility & experiment tracking
# ---------------------------------------------------------------------------

seed = 0
exp_root = os.path.join(_this_dir, 'output')
exp_name = 'cvpr_fv_run'
save_model = True
load_weight = ''

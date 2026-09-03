# Human-rating file format

The public scripts can regenerate the blank, stratified 300-claim annotation
sheet and recompute all reported agreement statistics. Store the original
anonymised filled CSV here with these columns:

```text
corpus,source_id,claim,pseudo_label,annotator_1,annotator_2,annotator_3
```

Then run:

```bash
python RESULTS/scripts/resolve_agreement.py \
  --input RESULTS/raw/human_eval/human_eval_filled.csv \
  --output-json RESULTS/raw/human_eval/human_eval_metrics.json
```

Keep the individual annotations exactly as collected rather than reconstructing
them from aggregate counts.

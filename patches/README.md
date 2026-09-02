# Det2Ver compatibility patch

`det2ver_conflict_export.patch` adds the per-instance fields required by the
spurious-conflict experiment to the public Det2Ver implementation. Apply it to
a clean sibling clone before checkpoint-mode reproduction:

```bash
git clone https://github.com/albert-jin/Det2Ver.git ../Det2Ver
git -C ../Det2Ver apply ../CVPR-fv/patches/det2ver_conflict_export.patch
```

The patch does not change training or prediction rules. It only preserves the
source FEVER ID/claim and writes `predictions.jsonl` during validation.

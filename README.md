# WDPS
python=3.8

#
Dataset preparation

## Download the dataset

### CoNLL-2003

```shell

```

### TACRED

### TACREV

#### Dev Split

```bash
python scripts/apply_tacred_patch.py \
  --dataset-file <TACRED DIR>/dev.json \
  --patch-file ./patch/dev_patch.json \
  --output-file ./dataset/dev_rev.json
```
#### Test Split

```bash
python scripts/apply_tacred_patch.py \
  --dataset-file <TACRED DIR>/test.json \
  --patch-file ./patch/test_patch.json \
  --output-file ./dataset/test_rev.json
```

### BoolQ


### 
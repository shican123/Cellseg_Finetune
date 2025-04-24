# CellSeg Fine-Tune

A lightweight fine-tuning framework for cell segmentation, supporting:

- **Cellpose**
- **CellBin v3**

Supports multiple image types (e.g., HE, SS), with options for training from scratch or using pretrained models.

---

## Requirements

To install dependencies (recommended in a virtual environment):

```bash
pip install -r requirements.txt
```

## Prepare Your Data

### Format

Images should be `.tif`.

Masks format for `Cellpose` can only be `instance mask`.

If necessary, use this tool to convert mask formats:

```bash
# Instance -> Semantic
python tools/mask_convert.py -i input_folder -o output_folder -m i2s

# Semantic -> Instance
python tools/mask_convert.py -i input_folder -o output_folder -m s2i
```

## Generate a Training List

```bash
python tools/make_trainset_txt.py \
  --img_dir dataset/images \
  --mask_dir dataset/masks \
  --output_txt trainset_list/my_trainset_list.txt
```

This txt file will be used as a necessary parameter input for subsequent fine-tuning.

## Start Fine-Tuning

### Run the main script:

```bash
python run_finetune.py \
  -m cellpose \
  -t ss \
  -f trainset_list/my_trainset_list.txt \
  -p cyto \
  -r 0.9 \
  -b 8 \
  -v 16 \
  -e 100
  ```

### Parameter Description

| Parameter | Description |
|:----:|:----------:|
|  -m   | Model name: cellpose or v3 |
|  -t   | Image type: ss or he |
|  -f   | Path to .txt training list     |
|  -p   | Pretrained model path/name or scratch     |
|  -r   | Train/validation split ratio     |
|  -b   | Training batch size     |
|  -v   | Validation batch size    |
|  -e   | Number of training epochs     |

## Output

After training, outputs are saved to `finetuned_models/model_timestamp/`:

`.pth or .hdf5`: Finetuned model

`train_log.json`: Training/validation loss log

`loss_curve.png`: Training/validation loss plot

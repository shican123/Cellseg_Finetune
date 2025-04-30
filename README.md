# CellSegAdapt

A lightweight framework for fine-tuning cell segmentation models with small datasets, optimized for spatial transcriptomics (CellBin). Efficiently adapt pre-trained models to new tissue types or staining conditions using minimal annotations. Achieve higher accuracy in cell boundary detection with just a few training samples. Supporting:

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

Images and masks format should be `.tif`.

Masks format for `Cellpose` can only be `instance mask`.

If necessary, use `tools/mask_convert.py` to convert mask formats:

```bash
# Instance -> Semantic
python tools/mask_convert.py -i input_folder -o output_folder -m i2s

# Semantic -> Instance
python tools/mask_convert.py -i input_folder -o output_folder -m s2i
```

### Naming Requirements

The only difference between the file names of each image and its corresponding mask is the suffix `"-img.tif"` and `"-mask.tif"`.

### Image Size

The input image size of the fine-tuning model is fixed at `256×256`. If necessary, use `tools/crop_chips.py` to crop the chip image to a smaller image of a specified size.

```bash
python tools/crop_chips.py \
  -i path/to/raw_images \
  -o path/to/output_patches \
  -s 256
```
The default value of the -s parameter is 256.

Input file format must be `.tif`, and the suffixes must be `"-img.tif"` or `"-mask.tif"`.

If filename contains coordinates like `chipA-x512_y512_w256_h256-img.tif,` cropping will retain correct spatial origin.

Output format: `chip_id-xX_yY_wW_hH-img/mask.tif`, useful for traceability.

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

### Required Parameter

| Parameter | Description |
|:----:|:----------:|
|  -m   | Model name: `cellpose` or `v3` |
|  -t   | Image type: `ss` or `he` |
|  -f   | Path to `.txt` training list     |
|  -p   | Pretrained model path/name or scratch  |

### Optional Parameter

| Parameter | Default | Description |
|:----:|:----------:|:----------:|
|  -r   | 0.9 |  Train/validation split ratio     |
|  -b   | 6  |Training batch size     |
|  -v   | 16 |Validation batch size    |
|  -e   | 500 | Number of training epochs. For the v3 model, due to its early stopping mechanism, it is the maximum number of training rounds. For the Cellpose model without early stopping, Cellpose officially recommends training for 100 epochs, and it may help to use more epochs, especially when you have more training data.     |

## Output

After training, outputs are saved to `finetuned_models/model_timestamp/`:

`.pth or .hdf5`: Finetuned model

`train_log.json`: Training/validation loss log

`loss_curve.png`: Training/validation loss plot

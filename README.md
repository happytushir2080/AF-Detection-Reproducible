# ECG AF Detection Model Training Code

This repository contains the model training code for the paper **“XXX”**.
The training pipeline uses ECG records from the MIT-BIH Atrial Fibrillation Database and converts them into fixed-length 10-second chunks before training the model.

The repository includes preprocessing code and Kaggle notebooks for training using two different splitting strategies:

1. **Patient-wise split**
2. **Random split**

---

## Overview

The complete training workflow is:

1. Download/prepare MIT-BIH AF ECG records.
2. Convert full ECG records into 10-second chunks.
3. Upload the processed data and training notebook to Kaggle.
4. Run model training using one of the provided notebooks.
5. Save the trained model weights for inference.

---

## File Description

| File                                         | Description                                                                            |
| -------------------------------------------- | -------------------------------------------------------------------------------------- |
| `convert-mit-bit-af.py`                      | Converts MIT-BIH AF ECG records into fixed-length 10-second chunks for model training. |
| `implementation-mit-bih-patient-split.ipynb` | Kaggle notebook for model training using patient-wise train/test split.                |
| `implementation-mit-bih-random-split.ipynb`  | Kaggle notebook for model training using random train/test split.                      |

> Note: The file `convert-mit-bit-af.py` is used for MIT-BIH AF data conversion.

---

## Data Conversion

Before training, the ECG records need to be converted into 10-second windows.

Run:

```bash
python convert-mit-bit-af.py \
  --in_dir "path-to-input-folder" \
  --out_dir "path-to-output-folder" \
  --win_sec 10.0
```

### Arguments

| Argument    | Description                                                            |
| ----------- | ---------------------------------------------------------------------- |
| `--in_dir`  | Folder containing MIT-BIH AF files such as `.hea`, `.dat`, and `.atr`. |
| `--out_dir` | Output folder where the converted 10-second chunks will be saved.      |
| `--win_sec` | Window length in seconds. Default value is `10.0`.                     |

Example:

```bash
python convert-mit-bit-af.py \
  --in_dir /path/to/mit-bih-af \
  --out_dir /path/to/mit-bih-af-10sec \
  --win_sec 10.0
```

---

## Model Training

The final model training is performed on **Kaggle**.

Upload the following to Kaggle:

1. The processed 10-second ECG chunk dataset.
2. One of the training notebooks:

   * `implementation-mit-bih-patient-split.ipynb`
   * `implementation-mit-bih-random-split.ipynb`

Then run the selected notebook on Kaggle.

---

## Training Notebooks

### 1. Patient-wise Split Training

Use:

```text
implementation-mit-bih-patient-split.ipynb
```

This notebook performs model training using a patient-wise split.
In this setting, ECG chunks from the same patient are not shared between training and testing sets.

This split is useful for evaluating the generalization ability of the model on unseen patients.

---

### 2. Random Split Training

Use:

```text
implementation-mit-bih-random-split.ipynb
```

This notebook performs model training using a random split of ECG chunks.

This split is useful for comparison, but it may include ECG chunks from the same patient in both training and testing sets.

---

## Saved Models

After training, save the final trained model weights from Kaggle.

The saved model files used for inference are available here:

[Download Saved Models](https://drive.google.com/drive/folders/1246Q1TwHK3CLAsEC9tCGY1bgXXHQFtXa?usp=sharing)

---

## Complete Training Pipeline

```bash
# 1. Convert MIT-BIH AF records into 10-second chunks
python convert-mit-bit-af.py \
  --in_dir /path/to/mit-bih-af \
  --out_dir /path/to/mit-bih-af-10sec \
  --win_sec 10.0
```

After this step:

1. Upload the output dataset folder to Kaggle.
2. Upload the required `.ipynb` training notebook.
3. Run the notebook.
4. Download/save the trained model weights.
5. Use the saved model weights with the inference repository.

---

## Requirements

The preprocessing script requires Python and common ECG/data-processing libraries.

Install dependencies using:

```bash
pip install -r requirements.txt
```


---

## License

**License:** Proprietary — © IIT Delhi; permitted only for non-commercial academic and research use with proper citation.





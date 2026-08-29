# Local Windows setup

The tested local layout is:

```text
D:\DLSS\satellite-disaster-damage-mapping\
  .venv\
  data\
    splits.csv
    splits_summary.csv
  xbd\
    hold\{images,labels,masks}
    test\{images,labels,masks}
    tier1\{images,labels,masks}
    tier3\{images,labels,masks}
```

## Environment

From PowerShell in the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m ipykernel install --user --name xbd-local --display-name "Python (xbd-local)"
```

Verify the GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The tested machine reports PyTorch `2.11.0+cu128`, CUDA available, and an
NVIDIA GeForce RTX 4070 Laptop GPU.

## Kaggle credentials and xBD download

Put the Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`. Do not add it
to this repository.

The official one-stream Kaggle download is slow on the tested connection. The
repository therefore includes an eight-range downloader that resumes each
part independently:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_xbd.py --parts 8 --workers 1
```

Re-run the same command after an interruption. Completed bytes in
`.download-parts\` are reused. When all parts finish, the script creates
`xbd-dataset.zip`.

Extract the archive into the repository root. Its top-level `xbd\` directory
must sit in the repository root, next to the `data\` directory:

```powershell
tar.exe -xf .\xbd-dataset.zip
```

Then validate the data and split assignment without overwriting the supplied
CSV files:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_splits.py --src .\xbd --out .\data --dry-run
```

## Run locally

Start Jupyter from the repository root so relative paths resolve correctly:

```powershell
.\.venv\Scripts\jupyter-lab.exe
```

Run `notebooks\data_cleaning_eda.ipynb` first. Its patch-extraction section
creates `work-pretrained\patches\manifest.csv`, `pre.npy`, and `post.npy`.
Both model notebooks read these exact arrays.

Next open `notebooks\pretrained_resnet18_train.ipynb`. Its default
`RUN_TRAINING = False` mode downloads or loads the published checkpoint and
recomputes validation, test-ID, and wildfire test-OOD metrics. Set
`RUN_TRAINING = True` for the full training schedule: five frozen-backbone
epochs followed by up to 45 fine-tuning epochs, with early stopping on
validation macro-F1.

The pretrained run writes generated checkpoints, metrics, and figures under
`work-pretrained\`. The tracked reference results under
`results\pretrained_resnet18\` are not overwritten.

Train the from-scratch CNN on the same extracted patches with:

```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute `
  .\notebooks\train_scratch_cnn.ipynb `
  --output train_scratch_cnn.executed.ipynb `
  --output-dir .\notebooks `
  --ExecutePreprocessor.kernel_name=xbd-local `
  --ExecutePreprocessor.timeout=-1
```

Use the scratch notebook configuration cell to set `RESUME = True` after an
interrupted scratch-CNN run. The pretrained notebook always restores its best
validation checkpoint after training.

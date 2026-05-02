# VS Code + Codex + Google Colab Workflow

This repo can use Codex locally in VS Code while Google Colab supplies the compute.

Recommended flow:

1. Use Codex/VS Code to edit files locally.
2. Commit and push changes to GitHub.
3. In Colab, pull the latest commit.
4. Run training/experiments on Colab compute.
5. Save heavy outputs to Google Drive, not Git.

Repo remote:

```bash
https://github.com/Rajshree-Mittal/01.git
```

## 1. Colab Setup Cell

Open Google Colab, choose `Runtime > Change runtime type`, then select a GPU or TPU if needed.

Run this cell first:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then clone or update the repo:

```python
import os

REPO_URL = "https://github.com/Rajshree-Mittal/01.git"
REPO_DIR = "/content/01"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    %cd {REPO_DIR}
    !git pull origin main

%cd {REPO_DIR}
```

## 2. Install Dependencies

This repo currently does not include a `requirements.txt`, so install only what your notebook/script needs.

Common ML stack:

```python
!pip install -q numpy pandas scikit-learn matplotlib seaborn xgboost imbalanced-learn
```

If you later add a `requirements.txt`, use:

```python
!pip install -r requirements.txt
```

## 3. Run Repo Code In Colab

Examples:

```python
!python data_balancing.py
```

```python
!python testing_data_Xgboost.py
```

```python
!python comparision.py
```

For the notebook:

```python
%run pida_experimentation_analysis.ipynb
```

## 4. Codex/VS Code Side

After Codex makes code changes locally:

```bash
git status
git add .
git commit -m "Update experiment code"
git push origin main
```

Then in Colab:

```python
%cd /content/01
!git pull origin main
```

## 5. Saving Colab Outputs

Keep large datasets, checkpoints, model weights, and generated outputs in Google Drive:

```python
OUTPUT_DIR = "/content/drive/MyDrive/pida_outputs"
!mkdir -p "{OUTPUT_DIR}"
```

Example:

```python
model_path = "/content/drive/MyDrive/pida_outputs/model.pkl"
```

Do not commit large generated files unless they are intentionally part of the repo.

## 6. If You Need To Push From Colab

Only do this for small code/notebook changes. Prefer editing with Codex locally.

In Colab:

```python
!git config --global user.name "Rajshree Mittal"
!git config --global user.email "your-email@example.com"
```

For private repos, use a GitHub personal access token or SSH key. Do not paste tokens into shared notebooks. Use Colab secrets when possible.

## 7. Mental Model

Codex does the coding in your local VS Code project.

Colab does the compute after pulling from GitHub.

GitHub is the sync point between them.

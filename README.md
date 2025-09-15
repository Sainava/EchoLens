# EchoLens: Toxicity GNN + Community Health Analysis

This workspace contains a Jupyter-driven pipeline to:
- Download or load a YouTube comments dataset with toxicity scores/labels
- Train a heterogeneous GNN (users ↔ comments) for toxic comment classification
- Classify reply edges as abusive vs non-abusive
- Analyze conversation motifs and k-core structure for “gang-up” indicators
- Build a user–user graph, detect communities, and compute polarization metrics

The main entrypoint is the notebook `gnn_training.ipynb` in the repository root.

## Contents
- `Notebooks/download_datasets.ipynb`: optional helper to fetch/prepare datasets
- `gnn_training.ipynb`: end-to-end GNN training, edge classification, motifs/k-core, and community metrics
- `requirements.txt`: minimal base dependencies (add the ML/GNN packages below)
- `artifacts/`: outputs (created at runtime)

## Data assumptions
Place a CSV under `Notebooks/` with a name like `youtube_comments_with_toxicity_<...>.csv`. The training notebook will attempt to discover it automatically via `context.md`, or it will scan the `Notebooks/` folder.

Required and optional columns (case-sensitive):
- Required
  - `AuthorChannelID`: user identifier
  - `CommentText`: text content of the comment
  - `CommentID`: unique identifier for the comment
  - One of the toxicity signals:
    - Either `ToxicLabel` (e.g., "toxic" vs "clean"), or
    - `ToxicScore` (float in [0,1])
- Optional (recommended)
  - `ParentCommentID`: for explicit reply edges (comment → child comment)
  - `VideoID`, `PublishedAt`: used to heuristically infer reply order if `ParentCommentID` is missing

The notebook creates a binary target column `ToxicBinary` if not present:
- If `ToxicLabel` exists and is a string, it marks rows starting with "toxic" as 1, else 0
- Otherwise, it derives `ToxicBinary = (ToxicScore > 0.7)` (you can adjust this threshold in the notebook)

## Environment setup
You can use either conda or venv. Below are typical steps for macOS (zsh).

1) Create and activate an environment

```zsh
# Conda (recommended)
conda create -n fafo python=3.12 -y
conda activate fafo

# or venv
python3 -m venv .venv
source .venv/bin/activate
```

2) Install Python packages

`requirements.txt` only lists a minimal base; install the ML/GNN stack as well:

```zsh
# Base
pip install -r requirements.txt

# ML / GNN / NLP stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install scikit-learn sentence-transformers networkx
```

Notes:
- On Apple Silicon, torch may use Metal (MPS) acceleration automatically; the notebook also supports CUDA if available.
- If `torch-geometric` needs extra wheels for your platform, see: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

3) Jupyter kernel

```zsh
python -m ipykernel install --user --name fafo --display-name "Python (fafo)"
```
In VS Code, pick the kernel "Python (fafo)" before running notebooks.

## Running the pipeline
1) Optional: fetch/prepare data
- Open `Notebooks/download_datasets.ipynb` and run the cells you need, or place your CSV manually under `Notebooks/`.

2) Train and analyze
- Open `gnn_training.ipynb` and Run All cells.
- The notebook will:
  - Load and clean the dataset
  - Build `HeteroData` with node types `user` and `comment`, and edges:
    - `('user','authored','comment')`
    - `('comment','replies_to','comment')` (from `ParentCommentID` or a time-ordered heuristic per `VideoID`)
  - Compute text embeddings (Sentence-Transformers) or fall back to hash-based vectors if model download fails
  - Train a Heterogeneous GNN (SAGEConv) for toxic comment classification with early stopping
  - Evaluate on test split and save the best checkpoint
  - Classify reply edges as abusive vs non-abusive using an MLP over concatenated parent/child comment embeddings
  - Convert the comment–comment reply graph to NetworkX and compute motifs, clustering, reciprocity, and k-core
  - Build a user–user graph from replies, detect communities (greedy modularity), compute modularity and E–I index

## Models and methods
- GNN: `HeteroConv` with `SAGEConv` per relation, mean aggregation, hidden size configurable (default 128)
- Node task: comment toxicity (2 classes) with cross-entropy; macro-F1 monitored for early stopping
- Edge task: MLP classifier on `[parent_emb || child_emb]` for reply edges (label = child toxic vs not)
- Motifs/k-core: uses NetworkX
  - Metrics: density (undirected), average clustering, triadic census (fallback to triangle count), reciprocity, max k-core
  - Reports both for all reply edges and for abusive-only (child toxic) subgraph
- User communities/polarization:
  - User–user directed edges from parent → child author
  - Communities on undirected projection via `greedy_modularity_communities`
  - Modularity (if computable) and E–I index on directed edges

## Outputs (artifacts)
Artifacts are written to `artifacts/` after running `gnn_training.ipynb`:
- `model_best.pt`: best model checkpoint (state dict)
- `config.json`: run configuration
- `edge_clf_report.json`: precision/recall/F1/support for abusive vs non-abusive reply edges
- `motifs_kcore_report.json`: graph metrics for all reply edges vs abusive-only edges
- `communities_report.json`: user–user graph stats: number of edges, communities, modularity, E–I index (all vs toxic-only)
- `user_partition.csv`: per-user community id for the full user–user graph
- `user_partition_toxic.csv`: per-user community id for the toxic-only user–user graph

## Tuning and customization
- Edit `gnn_training.ipynb` config cell to set:
  - `embedding_model` (e.g., `all-MiniLM-L6-v2`)
  - `gnn_hidden`, `epochs`, `lr`, `weight_decay`, `early_stopping_patience`
  - Train/val/test split ratios
  - Toxic threshold: update the logic that derives `ToxicBinary` from `ToxicScore`

## Troubleshooting
- Sentence-Transformers download fails:
  - The notebook falls back to hash-based vectors. To fix downloads, ensure internet access and install `sentence-transformers`.
- `torch-geometric` install:
  - Check the official installation guide for platform-specific wheels.
- Empty or no `ParentCommentID`:
  - The notebook builds a heuristic reply chain per `VideoID` by time order. If your data lacks `VideoID`/`PublishedAt`, you’ll have fewer or no reply edges.
- Small graphs:
  - Triadic census or modularity may fail on very small graphs; the notebook catches and falls back where possible.

## Reproducibility
- Seed is set for numpy, Python `random`, and torch; hardware backends (CUDA/MPS) may still introduce nondeterminism.

## Privacy note
- Ensure your dataset complies with platform terms and privacy regulations. Avoid committing raw data to version control.

## Project structure
```
FAFO/
├─ Notebooks/
│  └─ download_datasets.ipynb
├─ gnn_training.ipynb
├─ requirements.txt
└─ artifacts/        # created at runtime
```

## Quickstart
```zsh
# 1) Create env and install deps
conda create -n fafo python=3.12 -y && conda activate fafo
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric scikit-learn sentence-transformers networkx ipykernel
python -m ipykernel install --user --name fafo --display-name "Python (fafo)"

# 2) Place your CSV under Notebooks/
# 3) Open VS Code, select kernel "Python (fafo)", run gnn_training.ipynb
```

---

Completion summary
- Added a thorough README with setup, data schema, run steps, model/methods, and artifacts.

Requirements coverage
- README created (thorough): Done
- Included environment and run instructions for macOS/zsh: Done
- Documented outputs and analysis for Tasks A & B: Done

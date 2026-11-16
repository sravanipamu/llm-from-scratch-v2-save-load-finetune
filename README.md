# llm-from-scratch-v2-save-load-finetune

Learn how to **save**, **load** and **fine-tune** a small language model built from scratch in PyTorch.

> **Previous project:** [llm-from-scratch-v1-transformer-with-attention](https://github.com/sravanipamu/llm-from-scratch-v1-transformer-with-attention) builds and trains the model used here.
>
> **Next project:** [llm-from-scratch-v3-gpt2-pretrained-weights](https://github.com/sravanipamu/llm-from-scratch-v3-gpt2-pretrained-weights) builds the full GPT-2 model and loads OpenAI's pretrained weights.

---

## Why this project?

Training a model takes time. In v1, the model was trained every time you ran the notebook, and everything it learned was lost when the notebook closed.

Real projects work differently:

1. **Train once** and **save** what the model learned to a file.
2. **Load** that file later and use the model straight away, without training again.
3. **Fine-tune**: load the saved model and train it a little more on new data, instead of starting from zero.

This project shows all three steps on a tiny model, so each step is easy to follow.

---

## Key ideas (read this first)

| Term | Meaning in plain words |
|---|---|
| **Vocabulary (vocab)** | The list of every word the model knows. Each word gets a number (ID), for example `"He" → 1`, `"She" → 2`. |
| **Parameters / weights** | The numbers inside the model that change during training. They are what the model "learned". This model has **13,872** of them. |
| **`state_dict`** | A PyTorch dictionary that holds all the model's weights by name. This is what we save and load. |
| **Saving** | `torch.save(model.state_dict(), "models/parameters.bin")` writes the weights to a file. |
| **Loading** | `model.load_state_dict(torch.load("models/parameters.bin"))` copies the saved weights back into a model. |
| **Fine-tuning** | Starting from saved weights and training more on new sentences. The model keeps what it knew and learns something new. |

**Important:** a weights file holds only numbers, not code. To load it, you must first create the **same model class** (`GPTModel`) with the **same sizes**. That is why every notebook defines the model again.

---

## The model

The model is the small GPT-style model from v1:

- 70-word vocabulary built from 16 training sentences
- Each word is turned into a 24-number vector (embedding)
- 6 Transformer blocks with self-attention
- Reads up to 6 words at a time and predicts the next word

Training sentences look like this:

```
She composes songs and practices piano daily.
He reads books and explores the nearby caves.
```

---

## Project structure

```
llm-from-scratch-v2-save-load-finetune/
├── 1-save-model.ipynb    Step 1: train the model and save it
├── 2-load-model.ipynb    Step 2: load the saved model and generate text
├── 3-fine-tune.ipynb     Step 3: load the saved model and train it on new sentences
├── requirements.txt      Python packages to install
└── models/               Created when you run Step 1 (not in Git)
```

The `models/` folder is **not** in this repository. It is listed in `.gitignore` because the notebooks create it when you run them.

---

## Setup (one time)

### 1. Check that Python is installed

You need Python 3.10 or newer.

```bash
python3 --version
```

### 2. Download this repository

```bash
git clone https://github.com/sravanipamu/llm-from-scratch-v2-save-load-finetune.git
cd llm-from-scratch-v2-save-load-finetune
```

### 3. Create a virtual environment (recommended)

A virtual environment keeps this project's packages separate from other projects.

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

You will see `(.venv)` at the start of your terminal line when it is active.

### 4. Install the packages

```bash
pip install -r requirements.txt
```

This installs **PyTorch** (to build and train the model) and **Jupyter Notebook** (to run the `.ipynb` files).

### 5. Open the notebooks

```bash
jupyter notebook
```

A browser tab opens with the project files. You can also open the notebooks in VS Code with the Jupyter extension.

To run a notebook, open it and run every cell from top to bottom (**Run → Run All Cells**, or press `Shift + Enter` on each cell).

---

## Step 1: Train and save the model

**Notebook:** [1-save-model.ipynb](1-save-model.ipynb)

What happens, in order:

1. **Prepare the data.** The 16 sentences are split into words. A space is added before each `.` so the full stop counts as its own word.
2. **Build the vocabulary.** Every unique word is collected and sorted. This gives 70 words.
3. **Save the vocabulary** to `models/vocab.json`. The model only works with numbers, so later notebooks need this file to turn words into the same numbers again.
4. **Create training examples.** Each input is 6 word IDs, and the target is the same 6 IDs shifted one place to the left. The model learns to predict the next word.
   ```
   Input : She composes songs and practices piano
   Target: composes songs and practices piano daily
   ```
5. **Build the model** (`Attention` → `TransformerBlock` → `GPTModel`) and count its parameters: **13,872**.
6. **Train** for 100 epochs. Watch the `loss` value go down; that means the model is learning.
7. **Test** by generating text:
   ```
   He reads     → He reads books and explores the
   She composes → She composes songs and practices piano
   ```
8. **Save the weights** to `models/parameters.bin`:
   ```python
   torch.save(model.state_dict(), os.path.join(save_dir, "parameters.bin"))
   ```

**Files created:**

```
models/vocab.json        the 70 words, in order
models/parameters.bin    the trained weights
```

---

## Step 2: Load the model (no training)

**Notebook:** [2-load-model.ipynb](2-load-model.ipynb)

> Run Step 1 first. This notebook needs `models/vocab.json` and `models/parameters.bin`.

What happens, in order:

1. **Load the vocabulary** from `models/vocab.json` and rebuild the word ↔ ID mappings.
2. **Define the same model classes** as in Step 1. They must match exactly, or the weights will not fit.
3. **Create an empty model.** At this point its weights are random and it would produce nonsense.
4. **Load the saved weights** into it:
   ```python
   model.load_state_dict(torch.load(os.path.join(save_dir, "parameters.bin")))
   ```
5. **Generate text.** The output is the same as at the end of Step 1, but with **no training at all**:
   ```
   He reads     → He reads books and explores the
   She composes → She composes songs and practices piano
   ```

**What you learn:** the trained knowledge lives in the weights file. Load the file and the model is ready to use.

---

## Step 3: Fine-tune the model

**Notebook:** [3-fine-tune.ipynb](3-fine-tune.ipynb)

> Run Step 1 first. This notebook needs `models/vocab.json` and `models/parameters.bin`.

In the original data, **"She" composes** and **"He" reads**. Here we teach the model the opposite, using only 2 new sentences:

```
He composes songs and practices piano daily.
She reads books and explores the nearby caves.
```

What happens, in order:

1. **Load the vocabulary** from `models/vocab.json`. We reuse the same vocabulary so every word keeps its ID.
2. **Create the model and load the saved weights** from `models/parameters.bin`. The model starts with everything it learned in Step 1.
3. **Train for 100 epochs on the 2 new sentences.** The loss starts at about 3.8, much lower than the 5.6 in Step 1, because the model already knows these words.
4. **Test** the new behaviour:
   ```
   He composes → He composes songs and practices piano
   She reads   → She reads books and explores the
   ```
5. **Save the updated weights** to a **new** file, `models/updated-parameters.bin`, so the original `parameters.bin` stays unchanged.

**File created:**

```
models/updated-parameters.bin    the fine-tuned weights
```

**Note:** fine-tuning sentences can only use words already in the vocabulary. A new word such as `"guitar"` has no ID, so the notebook fails with a `KeyError`. Adding new words means changing the vocabulary size and the model, which is beyond this project.

---

## Summary

| Step | Loads | Trains? | Saves |
|---|---|---|---|
| 1. Save | nothing | Yes, 16 sentences | `vocab.json`, `parameters.bin` |
| 2. Load | `vocab.json`, `parameters.bin` | No | nothing |
| 3. Fine-tune | `vocab.json`, `parameters.bin` | Yes, 2 new sentences | `updated-parameters.bin` |

---

## Common problems

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: models/vocab.json` or `models/parameters.bin` | Step 1 has not been run yet | Run all cells in `1-save-model.ipynb` first |
| `RuntimeError: Error(s) in loading state_dict` | The model class or sizes (`VOCAB_SIZE`, `EMB_DIM`, `CONTEXT_LEN`) differ from Step 1 | Use exactly the same model code and sizes as Step 1 |
| `KeyError: 'some_word'` | A word is not in the vocabulary | Use only words from the original 16 sentences |
| `ModuleNotFoundError: No module named 'torch'` | Packages are not installed, or the virtual environment is not active | Activate `.venv` and run `pip install -r requirements.txt` |

---

## Credits

- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

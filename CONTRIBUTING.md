# 🤝 Contributing

Thank you for your interest in contributing to **small-code-models**! This project supports the research paper [*Evaluating Small-Scale Code Models for Code Clone Detection*](https://doi.org/10.48550/arXiv.2506.10995), and community contributions help make the findings more reproducible and extensible.

---

## 🐛 Reporting Bugs

If you encounter errors when running the scripts, loading datasets, or reproducing results, please [open a GitHub Issue](https://github.com/jorge-martinez-gil/small-code-models/issues/new/choose) using the **Bug Report** template.

Please include:
- The script name and dataset you were using (e.g., `bcb_detection_models/codebert-bcb-01.py`)
- The full error traceback
- Your Python version and GPU/CPU environment
- The versions of key packages (`torch`, `transformers`, `datasets`)

---

## ➕ Adding a New Model or Dataset

### Adding a new model
1. Pick an existing script from a relevant directory (e.g., `bcb_detection_models/codebert-bcb-01.py`) as a template.
2. Replace the model name, tokenizer class, and model class with the new model's equivalents from the HuggingFace Hub.
3. Name the new file following the existing convention: `<model>-<dataset>-01.py`.
4. Verify that the script runs end-to-end and reports F1 / Precision / Recall.
5. Open a Pull Request and fill out the checklist below.

### Adding a new dataset
1. Create a new directory at the repository root named after the dataset (e.g., `newdataset_clone_detection_models/`).
2. Add one script per model following the existing naming convention.
3. Document the dataset source and format in a short comment block at the top of each script.
4. Open a Pull Request and fill out the checklist below.

---

## 🎨 Code Style Conventions

- Follow [PEP 8](https://pep8.org/) for all Python code.
- Use descriptive variable and function names.
- Keep imports at the top of the file, grouped (standard library → third-party → local).
- Add docstrings to all public functions.
- Keep each script self-contained and runnable without extra configuration files.

---

## ▶️ Running the Existing Scripts

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/jorge-martinez-gil/small-code-models.git
   cd small-code-models
   pip install -r requirements.txt
   ```
2. Download the required dataset (see dataset links in the README).
3. Update the dataset path inside the script you want to run (the path variables are clearly marked near the top of each `main()` function).
4. Run the script:
   ```bash
   python bcb_detection_models/codebert-bcb-01.py
   ```

---

## ✅ Pull Request Checklist

Before opening a PR, please confirm:

- [ ] The new or modified script runs without errors.
- [ ] The script follows the existing naming convention.
- [ ] Code is PEP 8 compliant.
- [ ] Docstrings are present for all new functions.
- [ ] The PR description explains what was changed and why.
- [ ] No hardcoded private paths or credentials are included.
- [ ] If a new model is added, the README model table has been updated.
- [ ] If a new dataset is added, the README dataset list has been updated.

---

## 💬 Questions?

Feel free to [open an Issue](https://github.com/jorge-martinez-gil/small-code-models/issues) or start a [GitHub Discussion](https://github.com/jorge-martinez-gil/small-code-models/discussions) if you have any questions.

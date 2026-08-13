# HealtHeartPredictionSystem

A cardiovascular monitoring research project that reconstructs 12-lead ECG signals from PPG/oximetry-style input using a **Physics-Informed Graph Neural Network (PIGNN)**, paired with a rule-based clinical classifier for arrhythmia, bradycardia, tachycardia, and hypoxemia detection.

Built on real ECG data from PhysioNet's MIT-BIH Arrhythmia Database and PTB Diagnostic ECG Database.

> **Author:** Jesus Eduardo Escobar Meza — Tecnológico de Monterrey

---

## Project Overview

This project explores whether a graph neural network, structured around the heart's actual electrical conduction system (SA node → AV node → His-Purkinje network → myocardial segments), can reconstruct a full 12-lead ECG from a lower-dimensional physiological signal such as PPG/oximetry.

It combines two complementary approaches:

1. **Rule-based classification** (`main.py`) — a fast, interpretable clinical pipeline that flags bradycardia, tachycardia, arrhythmia, and hypoxemia directly from HR/RR-interval statistics.
2. **Deep learning reconstruction** (`Modelo2.ipynb`) — a physics-informed GNN that learns to map an oximetry-derived signal onto a full 12-lead ECG waveform, using a graph whose topology mirrors real cardiac conduction pathways.

This is an active research/engineering project, and the README below reflects its **current, honest state** — including a known data-leakage limitation in the present training input.

---

## Project Versions

| File | Description | Status |
|---|---|---|
| `main.py` | Rule-based clinical classifier (bradycardia, tachycardia, arrhythmia, hypoxemia) using simulated SpO₂ | Stable |
| `HeartModelling_PIGNN.ipynb` | Initial PIGNN prototype (heart mechanics model + first GNN attempt) | Superseded — has a known data-leakage flaw |
| `Modelo2.ipynb` | Corrected training pipeline for the PIGNN | **Active / current version** |
| `Analisis.ipynb` | Standalone analysis of training curves from `history.json`, independent of training runs | Active |

---

## Features

- Real ECG signal loading from MIT-BIH and PTB (via `wfdb`)
- R-peak detection, RR-interval and HRV computation
- Rule-based classification into:
  - **Bradycardia** — HR < 60 bpm (leve/moderada/severa by threshold)
  - **Tachycardia** — HR > 100 bpm
  - **Arrhythmia** — irregular rhythm via coefficient of variation, sudden HR changes, SDNN, and ectopic-beat ratio
  - **Hypoxemia** — SpO₂ < 95% (simulated in `main.py`; real SpO₂ integration planned)
- Graph-based cardiac conduction model (SA/AV/His-Purkinje/myocardial segments) with learned message passing
- Physics-informed GNN (PIGNN) trained end-to-end to reconstruct 12-lead ECG from PPG-like input
- Training with MSE + Pearson-correlation + smoothness + physiological regularization losses
- Checkpointing, resume-from-checkpoint, and early stopping
- Learning-curve visualization decoupled from training (`Analisis.ipynb`)
- LaTeX technical report (Overleaf) covering the full mathematical/architectural scope

---

## Technologies Used

- Python 3.10+
- PyTorch
- NumPy / SciPy
- WFDB (WaveForm DataBase Python package)
- Pandas
- Matplotlib
- Jupyter Notebooks

---

## Current Results & Known Limitations

Training on `Modelo2.ipynb` currently plateaus around:

- **PRD (Percent RMS Difference):** ~65–66% (clinically acceptable is generally < 24%)
- **Pearson correlation (val/train):** ~0.70–0.73

**Root cause identified:** the current PPG-like input is a *pseudo-PPG* derived directly from ECG channel 0 (plus its 1st/2nd derivatives), rather than a real, independently recorded PPG signal. This causes data leakage — the model can partially "cheat" by reconstructing Lead I almost directly from its own input, which is why Lead I shows an anomalously low PRD (~10.9%) compared to every other lead (58–94%). This gap is the clearest diagnostic signal that the current numbers are not yet a genuine reconstruction result.

**Because of this**, the model is **not** described as clinically validated anywhere in this repo, and no specific accuracy claims are made in the report or CV materials — only the honest, current numbers above.

### Planned fixes (see Roadmap)
Architectural tuning alone gives limited gains while the input remains pseudo-PPG. The highest-leverage next step is migrating to a dataset with **real, independently recorded PPG** (e.g., BIDMC, MIMIC-III, or CapnoBase from PhysioNet).

---

## System Requirements

- Python **3.10+**
- PyTorch, NumPy, SciPy, WFDB, Matplotlib
- Datasets downloaded from PhysioNet:
  - MIT-BIH Arrhythmia Database (v1.0.0)
  - PTB Diagnostic ECG Database (v1.0.0)
- ~2 GB disk space for the current datasets (more if BIDMC/MIMIC-III are added later)
- 4 GB+ RAM recommended; a GPU is optional but speeds up PIGNN training significantly
- **Windows users:** `DataLoader` must be created with `num_workers=0` (multiprocessing incompatibility). In Jupyter, use `argparse.parse_known_args()` instead of `parse_args()`.

---

## Installation

### Clone the repository
```bash
git clone https://github.com/your-username/HealtHeartPredictionSystem.git
cd HealtHeartPredictionSystem
```

### Install dependencies
```bash
pip install numpy pandas scipy torch wfdb matplotlib
```

### Get the datasets

**Datasets are not stored in this repository** (see [Data](#data) below). Download them from PhysioNet and place them at the repo root:

```
HealtHeartPredictionSystem/
├── mit-bih-arrhythmia-database-1.0.0/
├── ptb-diagnostic-ecg-database-1.0.0/
├── main.py
├── HeartModelling_PIGNN.ipynb
├── Modelo2.ipynb
└── Analisis.ipynb
```

```bash
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/ -P .
wget -r -N -c -np https://physionet.org/files/ptbdb/1.0.0/ -P .
```
*(Flatten/rename the resulting folders to match the structure above, or adjust the `root_dir` argument passed to the dataset loaders.)*

---

## Usage

### Rule-based classification
```python
from main import load_classified_datasets

df = load_classified_datasets(base_path='.', max_mitbih=50, max_ptb=50)
print(df.head())
```

### Train the PIGNN (Modelo2)
Open `Modelo2.ipynb` and run all cells, or extract `main()` and run:
```bash
python -c "from modelo2 import main; main()" \
  --data_root . \
  --epochs 80 \
  --batch_size 8 \
  --hidden_dim 128 \
  --output_dir checkpoints
```
Training resumes automatically from `checkpoints/best_model.pt` if it exists.

### Analyze training results
`Analisis.ipynb` reads `checkpoints/history.json` independently of any active training run, so you can monitor progress mid-training or revisit results later without re-running the model.

---

## Model Architecture (PIGNN)

- **Conduction graph:** nodes represent SA node, AV node, His bundle, left/right bundle branches, Purkinje terminals, atrial and ventricular myocardial segments — connected by edges carrying learned conduction delay, weight, and coupling type (electrical / tissue-spread / mechanical).
- **Node dynamics:** a `GraphGRUCell` per layer performs message passing + gated recurrent update per timestep, driven by an oximetry-derived local input signal.
- **ECG decoding:** each node projects to a 3D "electrical source" contribution, aggregated by anatomical coordinates (dipole-style) and projected to 12 leads, followed by a temporal GRU refinement head.
- **Loss:** weighted combination of MSE, `1 - Pearson correlation` (morphology), temporal smoothness, and physiological regularization on membrane-like and tension-like latent variables.

Full mathematical detail is documented in the accompanying Overleaf/LaTeX technical report.

---

## Data

### MIT-BIH Arrhythmia Database
### PTB Diagnostic ECG Database

**Citation (both datasets):**
```
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000).
PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.
Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.
```

Both datasets are distributed under PhysioNet's Open Data Commons Attribution License (ODC-BY 1.0). They are **not included in this repository** — see [Installation](#installation) for download instructions.

---

## Roadmap

- [ ] Run 3–5 additional training seeds and report mean ± SD (current results are single-run)
- [ ] Ablation runs: reduced early-stopping patience (8–10 epochs), increased dropout (0.05 → 0.1–0.15)
- [ ] **Migrate to real PPG data** (BIDMC / MIMIC-III / CapnoBase) — the primary lever for closing the PRD gap
- [ ] Add a classification head bridging the PIGNN pipeline with the rule-based `main.py` classifier
- [ ] Finish LaTeX report bibliography and resolve remaining citation references
- [ ] User interface / real-time inference (future)

---

## Reference Work

This project's physics-informed, graph-structured approach is inspired in part by large-scale cardiac foundation model research, notably the **Cardiac Sensing Foundation Model (CSFM)** by Xiao Gu and Prof. David Clifton (Oxford, Institute of Biomedical Engineering), published in *Nature Machine Intelligence*. CSFM is referenced as a large-scale counterpart to the more targeted, physics-informed approach taken here.

---

## Project Status

**Work in progress — active research project.**

### Completed
- [x] Real ECG/dataset loading (MIT-BIH, PTB)
- [x] Rule-based clinical classifier
- [x] Conduction-graph PIGNN architecture
- [x] End-to-end training pipeline with checkpointing
- [x] Root-cause diagnosis of current performance ceiling (pseudo-PPG leakage)

### In progress
- [ ] Multi-seed training runs for statistically credible metrics
- [ ] Real-PPG dataset migration
- [ ] LaTeX report finalization

---

## Authors

- **Jesus Eduardo Escobar Meza**  - Development

---

## License

*(To be defined)*
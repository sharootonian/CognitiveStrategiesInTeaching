# Cognitive Strategies In Teaching

This repository contains code and data for the paper:

**"Mentalizing and heuristics as distinct cognitive strategies in human teaching"**
by Sevan Harootonian, Thomas L. Griffiths, Yael Niv, and Mark K. Ho

**Preprint:** https://osf.io/preprints/osf/ys49u_v1

---

## Overview

This project investigates the cognitive strategies people use when teaching others. Through three behavioral experiments, we examine whether people rely on mentalizing or simpler heuristics when deciding what information to teach.

**Try the task online:** [Graph Teaching Task Demo](https://sharootonian.github.io/CognitiveStrategiesInTeaching/GraphTeachingTask_demo/)

---

## Repository Structure

```
CognitiveStrategiesInTeaching/
│
├── data/                           # Data files (see Data section below)
│   ├── raw/                        # Raw experimental data
│   ├── preprocessed/               # Processed data files
│   ├── sim/                        # Model simulation outputs
│   └── tasksetup/                  # Experimental trial configurations
│
├── functions/                      # Core Python modules
│   ├── fitting.py                  # Model fitting functions
│   ├── mentor.py                   # Bayesian teacher models
│   ├── max_utility_models.py      # Heuristic models (Q-values, path utility)
│   ├── features.py                 # Feature extraction for heuristics
│   ├── preprocessing.py            # Data preprocessing functions
│   ├── model_comparison.py         # Model comparison & visualization
│   ├── graphWorld.py               # MDP environment & visualization
│   ├── subgraphs.py                # Efficient subgraph enumeration
│   └── [other modules]             # Additional utilities
│
├── Preprocessing.ipynb             # Data preprocessing pipeline
├── Exp1_modeling.ipynb             # Experiment 1 modeling & fitting
├── Exp1_plots.ipynb                # Experiment 1 visualizations
├── Exp2_modeling.ipynb             # Experiment 2 modeling & fitting
├── Exp2_plots.ipynb                # Experiment 2 visualizations
├── Exp3_modeling.ipynb             # Experiment 3 modeling & fitting
├── Exp3_plots.ipynb                # Experiment 3 visualizations
├── supplementary_materials.ipynb   # Supplementary figures
├── Linear_mixed-effects.R          # Mixed-effects analysis in R
│
├── GraphTeachingTask_demo/         # Interactive task demo (GitHub Pages)
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher (tested with 3.10.5)
- R (optional, for mixed-effects models)
- At least 20GB RAM for full model simulations
- ~30GB disk space for data and model outputs

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sharootonian/CognitiveStrategiesInTeaching.git
   cd CognitiveStrategiesInTeaching
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Workflow

### Reproduce Paper Figures

All preprocessing, model simulations, and model fitting have already been run and saved in the repository. To reproduce the figures from the paper, simply run:

```bash
jupyter notebook Exp1_plots.ipynb
jupyter notebook Exp2_plots.ipynb
jupyter notebook Exp3_plots.ipynb
jupyter notebook supplementary_materials.ipynb
```

### Re-running the Full Pipeline (Optional)

If you want to re-run the preprocessing and modeling from scratch:

**Step 1: Unzip Raw Data**
- Unzip the raw data files in the `data/raw/` directory for each experiment

**Step 2: Preprocessing**
```bash
jupyter notebook Preprocessing.ipynb
```
Loads raw experimental data, computes model predictions for each trial, calculates teaching scores, and saves preprocessed data.

**Step 3: Model Fitting**
```bash
jupyter notebook Exp1_modeling.ipynb  # ~30 min, ~10GB RAM
jupyter notebook Exp2_modeling.ipynb  # ~43 min, ~20GB RAM
jupyter notebook Exp3_modeling.ipynb  # ~45 min, ~20GB RAM
```
**Warning:** Model simulations are computationally intensive and require 10-20GB RAM.

---

## Memory Requirements

- **Preprocessing:** ~2GB RAM
- **Model fitting:** ~4GB RAM
- **Full simulations:** 10-20GB RAM depending on experiment


# DigitAfrica Workshop 2026
## Identifying Epistemic Enclaves and Understanding Polarisation

This repository contains the hands-on material for the six-hour workshop described in the schedule PDF. The notebook structure now mirrors the actual workshop flow: two morning modules on segregation and enclave detection, two afternoon modules on polarisation dynamics and YSocial simulation analysis, plus one exercise notebook per module.

## Core learning outcome

By the end of the day, participants should be able to load social network data, identify enclave-like structures mathematically, and study how algorithmic bias can push those structures toward polarisation.

## Repository structure

```text
.
├── data/
│   ├── raw/                 # Demo graph files and YSocial-style SQLite database
│   └── processed/           # Derived tabular outputs used by the notebooks
├── notebooks/
│   ├── 00_environment_check.ipynb
│   ├── modules/            # Four workshop modules, aligned with the PDF schedule
│   ├── exercises/          # One lightweight exercise notebook per module
├── scripts/
│   └── build_workshop_materials.py
├── slides/
└── requirements.txt
```

## Workshop structure

### Morning session: The Anatomy of Segregation

| Time | Section | Notebook |
|------|---------|----------|
| 09:30-11:00 | Module 1. Foundations: Mapping the Terrain | `notebooks/modules/01_foundations_mapping_the_terrain.ipynb` |
| 11:00-11:15 | Coffee break | - |
| 11:15-12:45 | Module 2. Detecting Epistemic Enclaves | `notebooks/modules/02_detecting_epistemic_enclaves.ipynb` |

### Afternoon session: The Mechanics of Division

| Time | Section | Notebook |
|------|---------|----------|
| 12:45-13:45 | Lunch break | - |
| 13:45-15:15 | Module 3. Simulating Polarisation Dynamics | `notebooks/modules/03_simulating_polarisation_dynamics.ipynb` |
| 15:15-15:30 | Coffee break | - |
| 15:30-17:00 | Module 4. The Controlled Sandbox (YSocial + `ysights`) | `notebooks/modules/04_ysocial_sandbox.ipynb` |

### Exercises

Each section also has a short companion exercise notebook in `notebooks/exercises/`:

- `01_foundations_exercises.ipynb`
- `02_enclaves_exercises.ipynb`
- `03_dynamics_exercises.ipynb`
- `04_ysocial_sandbox_exercises.ipynb`

## Getting started

### Local use:

```bash
git clone https://github.com/GiulioRossetti/DigitAfrica_workshop_2026.git
cd DigitAfrica_workshop_2026

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_workshop_materials.py
jupyter lab
```

### SoBigData RI:

- Download the repository content locally
- Open the the RI portal at: https://sobigdata.eu/
- Register to the E-Infrastructure
- Enter the SoBigData Lab (second icon on the left in the page header)
- Upload the repository in the SoBigData Jupyter Lab

## Hands-on

Open the notebooks in this order:

1. `notebooks/00_environment_check.ipynb`
2. `notebooks/modules/01_foundations_mapping_the_terrain.ipynb`
3. `notebooks/modules/02_detecting_epistemic_enclaves.ipynb`
4. `notebooks/modules/03_simulating_polarisation_dynamics.ipynb`
5. `notebooks/modules/04_ysocial_sandbox.ipynb`

## Notes on the YSocial section

Module 4 assumes access to a YSocial simulation database: you can either use [Ysocial](https://y-not.social) to generate your own simulation data or download the sample simulation available [here](https://data.d4science.net/mQAFv) (data provided in a zip archive to be extracted)

## Requirements

- Python 3.10+
- `networkx`, `cdlib`, `ndlib`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyterlab`
- `ysights` for the YSocial analysis notebook

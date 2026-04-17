# DigitAfrica Workshop 2026
## Identifying Epistemic Enclaves and Understanding Polarisation

A one-day hands-on course covering the computational methods used to detect and analyse epistemic enclaves and online polarisation in social networks.

---

## Repository structure

```
.
├── data/
│   ├── raw/          # Original, unmodified datasets
│   └── processed/    # Cleaned / pre-processed datasets
├── notebooks/        # Jupyter notebooks (one per session)
├── slides/           # Presentation slides (PDF / PPTX)
└── requirements.txt  # Python dependencies
```

## Course outline

| Time | Session | Notebook |
|------|---------|----------|
| 09:00 – 09:30 | Welcome & setup | `00_setup.ipynb` |
| 09:30 – 10:30 | Introduction to polarisation | `01_introduction.ipynb` |
| 10:30 – 12:00 | Data exploration | `02_data_exploration.ipynb` |
| 13:00 – 14:00 | Network construction & analysis | `03_network_analysis.ipynb` |
| 14:00 – 15:00 | Community detection | `04_community_detection.ipynb` |
| 15:00 – 16:00 | Measuring polarisation | `05_polarisation.ipynb` |
| 16:00 – 17:30 | Epistemic enclaves | `06_epistemic_enclaves.ipynb` |

## Getting started

```bash
# 1. Clone the repository
git clone https://github.com/GiulioRossetti/DigitAfrica_workshop_2026.git
cd DigitAfrica_workshop_2026

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch JupyterLab
jupyter lab
```

Then open the notebooks in the `notebooks/` folder in order, starting with `00_setup.ipynb`.

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for the full list of packages.

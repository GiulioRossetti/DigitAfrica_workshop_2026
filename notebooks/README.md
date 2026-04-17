# Notebooks

The notebook tree is organized around the four sections in the workshop schedule, plus a short environment check and one exercise notebook per section.

## Recommended order

| Notebook | Role |
|----------|------|
| `00_environment_check.ipynb` | Verifies dependencies and generates the demo assets |
| `modules/01_foundations_mapping_the_terrain.ipynb` | Module 1. Network loading, metrics, core extraction, visual exploration |
| `modules/02_detecting_epistemic_enclaves.ipynb` | Module 2. Community detection and enclave evaluation |
| `modules/03_simulating_polarisation_dynamics.ipynb` | Module 3. Bounded confidence and algorithmic bias |
| `modules/04_ysocial_sandbox.ipynb` | Module 4. YSocial simulation analysis with `ysights` |

## Exercises

The `exercises/` directory contains four lightweight practice notebooks:

- `exercises/01_foundations_exercises.ipynb`
- `exercises/02_enclaves_exercises.ipynb`
- `exercises/03_dynamics_exercises.ipynb`
- `exercises/04_ysocial_sandbox_exercises.ipynb`

## Running the notebooks

```bash
pip install -r ../requirements.txt
python ../scripts/build_workshop_materials.py
jupyter lab
```

The notebooks are designed to run against the demo graph files and the included `ysocial_demo.sqlite` database, but Module 4 can be pointed to a real YSocial export by editing the `DB_PATH` variable.

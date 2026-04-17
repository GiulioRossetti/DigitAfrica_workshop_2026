# Data

This directory contains all datasets used during the workshop.

```
data/
├── raw/          # Original, unmodified datasets
└── processed/    # Cleaned and pre-processed datasets ready for analysis
```

## Raw data

Place original data files in `raw/`.  
These files should never be modified; treat them as read-only.

Typical formats: `.csv`, `.json`, `.graphml`, `.edgelist`

## Processed data

Place any data that has been cleaned, filtered, or transformed in `processed/`.  
Scripts/notebooks that produce these files should document the transformation steps.

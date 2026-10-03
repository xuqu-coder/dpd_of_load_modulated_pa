# Digital predistortion study

A study-oriented simulation of digital predistortion (DPD) for a load-modulated power amplifier. The amplifier model is synthetic. It follows the paper's algorithmic idea and does not reproduce measured hardware results.

## Citation

Mohammad Abdollah Chalaki, Emma Gu, Ahmed Ben Ayed, Patrick Mitran, and Slim Boumaiza, "Robust Digital Predistortion of Load-Modulated Power Amplifiers Under Varying Signal Conditions," IEEE Microwave Magazine, 2026.

The publisher PDF is not included in this repository.

## Contents

- `study_dpd.ipynb` — the study notebook
- `dpd_utils.py` — waveform, basis, and learning helpers
- `requirements.txt` — Python packages used to run the notebook

## Setup

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Open `study_dpd.ipynb` after the packages are installed.

# AlphaFold's Bayesian Roots in Probability Kinematics

Supplementary code for the AISTATS 2026 paper:

**AlphaFold's Bayesian Roots in Probability Kinematics**  
Thomas Hamelryck, Kanti V. Mardia

---

## Overview

This repository contains code for the **synthetic von Mises random walk (VRW) experiment** described in the paper.

The experiment demonstrates **probability kinematics (PK)** as a generalized Bayesian update of a directional prior on angles using soft evidence on distances.

---

## What the code does

The script:

- Defines a 2D **von Mises random walk (VRW)** prior over angles  
- Computes the induced distribution over distances  
- Applies a **PK update via reference ratio**  
- Runs a **naive ablation (without reference distribution)**  
- Evaluates results using **Kolmogorov–Smirnov (KS) tests**  
- Generates:
  - `vrw_densities.png`
  - `vrw_trajectories.png`

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the experiment

```bash
python run_vrw_experiment.py
```

Optional: specify a random seed

```bash
python run_vrw_experiment.py 42
```

---

## Output

- Density comparison plot (`vrw_densities.png`)
- Trajectory visualization (`vrw_trajectories.png`)
- KS statistics printed to console

---

## Reproducibility

- Supports fixed random seeds  
- Runs 10 independent simulations  
- Reports min / median / max KS statistics  

---

## Citation

If you use this code, please cite:

```bibtex
@article{hamelryck2026alphafold,
  title={AlphaFold's Bayesian Roots in Probability Kinematics},
  author={Hamelryck, Thomas and Mardia, Kanti V.},
  journal={Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

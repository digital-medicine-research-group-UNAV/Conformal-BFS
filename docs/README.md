# Bidirectional Floating Feature Selection Guided by Uncertainty Quantification

**Authors:**  [Marcos López-De-Castro](https://github.com/MarcosLDC), [José González-Gomariz](), [Alberto García-Galindo]() [Farnoosh Abbas-Aghababazadeh](https://github.com/RibaA), [Kewei Ni](https://github.com/Nicole9801), [Benjamin Haibe-Kains](), [Ruben Armañanzas Arnedillo](https://github.com/rarmananzas), 

**Contact:** [mlopezdecas@unav.es](mailto:mlopezdecas@unav.es)

**Description:** Conformal prediction–driven feature selection, with applications in immuno-oncology datasets. We propose a novel bidirectional floating algorithm for feature selection named Conformal Bidirectional Floating Search Algorithm (CBFS), in which the feature search is enhanced by information from the conformal prediction framework.

--------------------------------------

[![pixi-badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square)](https://github.com/prefix-dev/pixi)


![GitHub last commit](https://img.shields.io/github/last-commit/bhklab/conformal-ffs?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/bhklab/conformal-ffs?style=flat-square)
![GitHub pull requests](https://img.shields.io/github/issues-pr/bhklab/conformal-ffs?style=flat-square)
![GitHub contributors](https://img.shields.io/github/contributors/bhklab/conformal-ffs?style=flat-square)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/bhklab/conformal-ffs?style=flat-square)

## Set Up

### Prerequisites

Clone environment in pixi.toml with 
```bash
pixi install

```
or 

Install via:
```bash
pip install cbfs

```

### Usage

```python
from cbfs import ffs


data_path = "your_path.h5ad" # .csv are also supported
target_column =  "target-column-name"
run_id= 0 # equivalent to random seed (int)

ffs_instance = ffs.FloatingFeatureSelector(run_id=run_id, data_path=data_path, target_column=target_column)
experiment_result = ffs_instance.run_ffs(n_feat=10)


print("Experiment completed successfully!")

print("Selected features:", experiment_result)

coverage = ffs_instance.Empirical_coverage_
uncertainty = ffs_instance.Uncertainty_
certainty = ffs_instance.Certainty_

print(f"Empirical coverage: {coverage}")
print(f"Uncertainty: {uncertainty}")
print(f"Certainty: {certainty}")

all_results[run_id] = {"selected_features": experiment_result, "run_id": run_id, "empirical_coverage": coverage,
                    "uncertainty": uncertainty, "certainty": certainty}


```




<!-- ## Documentation

Click [here](https://bhklab.github.io/conformal-ffs) to view the full documentation. --> 


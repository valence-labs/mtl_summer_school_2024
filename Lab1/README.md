# Lab 1: Virtual Screening 

In this first lab, we will delve into the realm of virtual screening. Using datasets of 2D molecules, we will develop predictive models to assess inhibitory activity against a human kinase EGFR(Epidermal Growth Factor Receptor) protein. Building on concepts from lectures on molecular representation, scoring, and Graph Neural Networks (GNNs) for Chemistry, we will utilize `PyTorch`,`PyG`, `Scikit-learn`, and other libraries to create both GNN models and classical Random Forest models with molecular fingerprints. The ultimate objective is to screen a small commercial library and select **100** promising and **diverse** molecules with molecular weight between **280 and 400 Da** for further experimental investigation.

----
To run the tutorials, please ensure all dependencies below are installed. 
- `datamol`
- `molfeat`
- `splito`
- `scikit-learn`
- `pytorch`
- `pyG`

You can install those dependencies by 
```shell
conda env create -f env.yml
```

---

The tutorials are also available on Google Colaboratory:

- [00_introduction_colab](https://colab.research.google.com/github/valence-labs/mtl_summer_school_2024/blob/main/Lab1/00_introduction.ipynb)
- [01_lab1_code](https://colab.research.google.com/github/valence-labs/mtl_summer_school_2024/blob/main/Lab1/01_lab1_code.ipynb)
- [02_recap_colab](https://colab.research.google.com/github/valence-labs/mtl_summer_school_2024/blob/main/Lab1/02_recap.ipynb)


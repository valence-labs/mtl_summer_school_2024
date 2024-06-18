# Lab 4: Transcriptomics and Target Deconvolution 

In this fourth and last lab, we will look at the role of transcriptomics in drug discovery
especially relating to target deconvolution and identification of off target effects. 
We'll start with an overview of bio-assays in drug discovery and dive into the advantages
of transcriptomics. Then we'll dig into the public and highly scalable connectivity map L1000
data and utlize this to 1 derive a comparable embedding space that relates the effect of small molecule
and genetic perturbations, 2) identify potential primary and secondary targets of everolimus, a
known MTOR inhibitor.

---
To run the tutorials, please ensure all dependencies below are installed. 
- `cmapPy`
- `umap-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `torch`
- `sklearn`

You can install those dependencies by 
```shell
conda env create -f env.yml
```

---

The tutorials are available as follows:

- [01_lab4_code]()
Or on [colab](https://colab.research.google.com/drive/1k7AWbdAlfUEJbb0Lj6ZcO_bhrpClFICV#scrollTo=yNaHyGqWJ9WUr)

The slides are available [here](https://docs.google.com/presentation/d/1gJvF8BTWwivgFE5R2cDIu5XQqBsQtVaGZc__kxQ8X8A/edit#slide=id.p)

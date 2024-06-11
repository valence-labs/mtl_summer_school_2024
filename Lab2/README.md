# Lab 2 : Binding Affinity Prediction with ML-Based Docking

In this second lab, we will continue our virtual screening of EGFR binders by understanding what binding free energy is and how we can leverage it. Using datasets of 2D molecules from Lab1, we will develop predictive models to assess the affinity  aagainst our EGFR protein. We will build concepts on how to use molecular docking to predict the binding affinity of a ligand to a protein. We will also introduce the concept of active learning to improve the performance of our models by utilizing the docking as an oracle.

----
To run the tutorials, please ensure all dependencies  are installed. 

You can install those dependencies by 
```shell
mamba env create -n lab2 -f env.yml
conda activate lab2
```

For visualization of the docking results, you need to install `py3Dmol` by running the following command:
```shell
pip install py3Dmol
```

For better Active learning performance, you can install `modAL` by running the following command:
```shell
pip install modAL-python
```

---
The tutorial is available in the following links:
## [Google colab Link](https://colab.research.google.com/drive/17Ws93qdwwcnlpmsKTAmrb9uRvGmolQHA?usp=sharing)
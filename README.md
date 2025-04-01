## BioNER Benchmarks

This repository contains the code and data for the paper "Exploring BioNER Frontiers: An In-Depth
Evaluation". The paper evaluates 10 state-of-the-art biomedical named entity recognition (BioNER) systems on 8 benchmark datasets, including 4 new datasets. The results show that the performance of BioNER systems varies significantly across different datasets, and that the choice of evaluation metrics can greatly affect the reported performance.

To reproduce the results, please follow the instructions below.

Run the following command to install the required packages:

```bash
./init.sh
```

With this command, you will install the required packages and you will download the required submodules (VANER and AIONER).

Once you have installed the required packages, you can run the different notebooks in the `bioner-benchmarks` directory. The notebooks are organized by models, and each notebook contains the code to run the model on the different datasets. The notebooks are:

- `benchmark_aioner.ipynb`: This notebook contains the code to run the AIONER model on the different datasets.
- `benchmark_biober_pubmed.ipynb`: This notebook contains the code to run the BioBERT and BioBERT-PubMed models on the different datasets.
- `benchmark_vaner.ipynb`: This notebook contains the code to run the VANER model on the different datasets.

# Lab-Rotation
This repository contains the code which I used for my lab rotation for the Borgwardt group.
The goal of the project was to integrate information about the genome and the environment to better predict human height as a model phenotype.
This was done using data from the [UKBiobank Study](https://pubmed.ncbi.nlm.nih.gov/25826379/)

## Brief file description
*integrate_environment.py*: code used to create the dataloaders. It adds the environmental data and the SNP data

*gln_model.py* : my implementation of the global-linear-model from Sigurdsson et al.

*lightning_module.py* : Pytorch Lightning module used for training the model for the input data including environment

*lightning_module_SNP_only.py* Pytorch Lightning module used for training the model for the SNP only input data (uses another data loader)

*utily.py* : contains the code for the score transformation of height based on sex and age

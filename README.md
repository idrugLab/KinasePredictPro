## **Environment**

The most important python packages are:
- python == 3.7.10
- pytorch == 1.9.0+cu111
- tensorboard == 2.5.0
- rdkit == 2021.09.3
- scikit-learn == 1.0
- hyperopt == 0.2.7
- numpy == 1.19.5

## **Models**

You can download all the model files from [here](https://github.com/idrugLab/KinasePredictPro/releases/tag/models) and extract them to the root of the repository.

## **Predict**

Args:

  - smi: The SMILES string of the molecule you want to predict.
  - model_type: `0`: Voting-RF-RDKitDes+FP2+AtomPairs; `1`: The best model based on each kinase.

E.g.

`python predict.py 'CC(C)OC(=O)CC(=O)CSC1=C(C=C2CCCC2=N1)C#N' 1`

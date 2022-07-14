import pandas as pd
import deepchem as dc
from rdkit import Chem
from ofa_smi_eval import featurizer_ofa, load_model_ofa, predict_ofa
import pandas as pd
import numpy as np

df_models = pd.read_csv('./data/kinase_jobs.csv')


def load(model_name, target_name, seed):
    df_single_model_list = \
    df_models[(df_models['method'] == model_name) & (df_models['UniProt ID'] == target_name)].values[0]
    algorithm = df_single_model_list[1].split('_')[0]
    featurizer = '_'.join(df_single_model_list[1].split('_')[1:])
    featurizer_obj = featurizer_ofa(featurizer)
    best_hyperparams_list = eval(df_single_model_list[2])
    model = load_model_ofa(target_name,
                           algorithm,
                           featurizer,
                           best_hyperparams_list,
                           seed1=seed)
    return model, algorithm, featurizer, featurizer_obj


def get_predict_helper(smi, target_name, seed):
    mol = Chem.MolFromSmiles(smi)
    model1_name = 'RF_Morgan_rdkit'
    model2_name = 'RF_FP2_rdkit'
    model3_name = 'RF_PharmacoPFP_rdkit'
    model1, algorithm1, featurizer1, featurizer_obj1 = load(model1_name,
                                                            target_name,
                                                            seed=seed)
    model2, algorithm2, featurizer2, featurizer_obj2 = load(model2_name,
                                                            target_name,
                                                            seed=seed)
    model3, algorithm3, featurizer3, featurizer_obj3 = load(model3_name,
                                                            target_name,
                                                            seed=seed)

    if len(featurizer1.split('_')) != 2:
        dataset1 = dc.data.NumpyDataset(
            X=np.array([featurizer_obj1._featurize(mol)]))
    elif len(featurizer1.split('_')) == 2:
        dataset1_1 = dc.data.NumpyDataset(
            X=np.array([featurizer_obj1[0]._featurize(mol)]))
        dataset1_2 = dc.data.NumpyDataset(
            X=np.array([featurizer_obj1[1]._featurize(mol)]))
        dataset1 = dc.data.NumpyDataset(
            X=np.concatenate([dataset1_1.X, dataset1_2.X], axis=1))
    if len(featurizer2.split('_')) != 2:
        dataset2 = dc.data.NumpyDataset(
            X=np.array([featurizer_obj2._featurize(mol)]))

    elif len(featurizer2.split('_')) == 2:
        dataset2_1 = dc.data.NumpyDataset(
            X=np.array([featurizer_obj2[0]._featurize(mol)]))
        dataset2_2 = dc.data.NumpyDataset(
            X=np.array([featurizer_obj2[1]._featurize(mol)]))
        dataset2 = dc.data.NumpyDataset(
            X=np.concatenate([dataset2_1.X, dataset2_2.X], axis=1))
    if len(featurizer3.split('_')) != 2:
        dataset2 = dc.data.NumpyDataset(
            X=np.array([featurizer_obj3._featurize(mol)]))
    elif len(featurizer3.split('_')) == 2:
        dataset3_1 = dc.data.NumpyDataset(
            X=np.array([featurizer_obj3[0]._featurize(mol)]))
        dataset3_2 = dc.data.NumpyDataset(
            X=np.array([featurizer_obj3[1]._featurize(mol)]))
        dataset3 = dc.data.NumpyDataset(
            X=np.concatenate([dataset3_1.X, dataset3_2.X], axis=1))
    data_pred1 = predict_ofa(target_name,
                             algorithm1,
                             featurizer1,
                             model1,
                             test_dataset=dataset1,
                             seed2=seed)
    data_pred2 = predict_ofa(target_name,
                             algorithm2,
                             featurizer2,
                             model2,
                             test_dataset=dataset2,
                             seed2=seed)
    data_pred3 = predict_ofa(target_name,
                             algorithm3,
                             featurizer3,
                             model3,
                             test_dataset=dataset3,
                             seed2=seed)
    y = (data_pred1 + data_pred2 + data_pred3) / 3
    return {target_name: ['Voting-RF-RDKitDes+FP2+AtomPairs', y[0]]}

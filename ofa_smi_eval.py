from sklearn.metrics import roc_auc_score
import deepchem as dc
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import os
import pickle
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from openbabel import pybel
from PyBioMed.PyMolecule.fingerprint import CalculateFP2Fingerprint
from rdkit import Chem
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.typing import RDKitMol
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.Pharm2D import Generate
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem import AllChem
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.models.graph_models import GraphConvModel
from deepchem.models import GATModel, AttentiveFPModel
from deepchem.models.torch_models import MPNNModel
from chemprop.utils import load_checkpoint
from fpgnn.tool import load_model
import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import math


def score(y_true, y_pre):
    try:
        auc_roc_score = roc_auc_score(y_true, y_pre)
    except:
        return 'wrong', 'wrong', 'wrong', 'wrong', 'wrong', 'wrong', 'wrong', 'wrong', 'wrong', 'wrong', 'wrong'
    y_pred_print = [round(y, 0) for y in y_pre]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_print).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    q = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt(
        (tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    return tp, tn, fn, fp, se, sp, mcc, q, auc_roc_score, F1, BA


def hand_split(dataset, seed=0):

    def load_dataset(dataset):
        x_load, y_load = [], []
        for x, y, w, id in dataset.itersamples():
            x_load.append(x)
            y_load.append(int(y))
        y_load = np.array(y_load)[:, np.newaxis]
        df_zip = pd.DataFrame(zip(x_load, y_load), columns=['X', 'y'])
        df_pos = df_zip[df_zip['y'].isin([[1]])]
        df_neg = df_zip[df_zip['y'].isin([[0]])]
        dataset_pos = dc.data.NumpyDataset(X=df_pos['X'].tolist(),
                                           y=df_pos['y'].tolist())
        dataset_neg = dc.data.NumpyDataset(X=df_neg['X'].tolist(),
                                           y=df_neg['y'].tolist())
        return dataset_pos, dataset_neg

    dataset_origin_pos, dataset_origin_neg = load_dataset(dataset)
    splitter = dc.splits.RandomSplitter()
    amount = len(dataset_origin_pos.y)
    if amount >= 6:
        frac_train, frac_valid, frac_test = 0.8, 0.1, 0.1
    elif amount < 6:
        frac_train, frac_valid, frac_test = 0.6, 0.2, 0.2
    train_dataset_pos, valid_dataset_pos, test_dataset_pos = splitter.train_valid_test_split(
        dataset=dataset_origin_pos,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        seed=seed)
    amount = len(dataset_origin_neg.y)
    if amount >= 6:
        frac_train, frac_valid, frac_test = 0.8, 0.1, 0.1
    elif amount < 6:
        frac_train, frac_valid, frac_test = 0.6, 0.2, 0.2
    train_dataset_neg, valid_dataset_neg, test_dataset_neg = splitter.train_valid_test_split(
        dataset=dataset_origin_neg,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test,
        seed=seed)
    train_dataset = dc.data.NumpyDataset.merge(
        [train_dataset_pos, train_dataset_neg])
    valid_dataset = dc.data.NumpyDataset.merge(
        [valid_dataset_pos, valid_dataset_neg])
    test_dataset = dc.data.NumpyDataset.merge(
        [test_dataset_pos, test_dataset_neg])
    return train_dataset, valid_dataset, test_dataset


def rdkit_feature_dataset(dataset):
    print('start rdkit feat')
    imp = SimpleImputer(strategy="constant", fill_value=0)
    sel = VarianceThreshold(threshold=(.7 * (1 - .7)))
    selector = SelectPercentile(percentile=30)
    x_load, y_load = [], []
    for x, y, w, id in dataset.itersamples():
        x[x == np.inf] = np.nan
        x_load.append(x)
        y_load.append(int(y))
    X_min = np.min(x_load, axis=0)
    X_max = np.max(dataset.X, axis=0)
    x_load_1 = imp.fit_transform(x_load)
    x_load_1 = pd.DataFrame(x_load_1)
    x_load_2 = sel.fit_transform(x_load_1)
    var_list = sel.get_support(indices=True)
    x_load_3 = selector.fit_transform(x_load_2, y_load)
    sel_list = selector.get_support(indices=True)
    final_sel_list = np.array(var_list)[np.array(sel_list)]
    y_load = np.array(y_load)[:, np.newaxis]
    dataset_new = dc.data.NumpyDataset(X=x_load_3, y=y_load)
    transformer = dc.trans.MinMaxTransformer(transform_X=True,
                                             dataset=dataset_new)
    dataset_new = transformer.transform(dataset_new)
    x_final = pd.DataFrame(dataset_new.X)
    y_final = dataset.y.ravel()
    return dataset_new, x_load_3, x_final, y_final, final_sel_list


def featurizer_ofa(featurizer_choice):

    class FP2fingerprint(MolecularFeaturizer):

        def _featurize(self, mol: RDKitMol) -> np.ndarray:
            smi = Chem.MolToSmiles(mol)
            mol = pybel.readstring("smi", smi)
            fp_fp2 = CalculateFP2Fingerprint(mol)
            fp = [0] * fp_fp2[0]
            for idx in fp_fp2[1]:
                fp[idx] = fp_fp2[1][idx]
            return fp

    class AtomPairFeaturizer(MolecularFeaturizer):

        def _featurize(self, mol: RDKitMol) -> np.ndarray:
            fp = list(Pairs.GetHashedAtomPairFingerprint(mol))
            fp = np.asarray(fp, dtype=float)
            return fp

    class NoFeaturizer(MolecularFeaturizer):

        def _featurize(self, mol: RDKitMol) -> np.ndarray:
            smiles = AllChem.MolFromSmiles(mol)
            # fp = np.asarray(fp_pubcfp, dtype=float)
            return smiles

    class ErGFeaturizer(MolecularFeaturizer):

        def _featurize(self, mol: RDKitMol) -> np.ndarray:
            fp_phaErGfp = AllChem.GetErGFingerprint(mol,
                                                    fuzzIncrement=0.3,
                                                    maxPath=21,
                                                    minPath=1)
            # fp = np.asarray(fp_pubcfp, dtype=float)
            return fp_phaErGfp

    class Pharmacophore2DFeaturizer(MolecularFeaturizer):

        def _featurize(self, mol: RDKitMol) -> np.ndarray:
            fp = list(Generate.Gen2DFingerprint(mol, sigFactory))
            fp = np.asarray(fp, dtype=float)
            return fp

    if featurizer_choice == 'Morgan':
        return dc.feat.CircularFingerprint(size=1024)

    if featurizer_choice == 'FP2':
        return FP2fingerprint()

    if featurizer_choice == 'MACCS':
        return dc.feat.MACCSKeysFingerprint()

    if featurizer_choice == 'AtomPairs':
        return AtomPairFeaturizer()

    if featurizer_choice in ['GAT']:
        return dc.feat.MolGraphConvFeaturizer()

    if featurizer_choice in ['AttentiveFP', 'MPNN']:
        return dc.feat.MolGraphConvFeaturizer(use_edges=True)

    if featurizer_choice == 'GCN':
        return dc.feat.ConvMolFeaturizer()

    if featurizer_choice in ['PharmacoPFP', 'PharamacoPFP']:
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        sigFactory = SigFactory(featFactory, minPointCount=2, maxPointCount=3)
        sigFactory.SetBins([(1, 3), (3, 8)])
        sigFactory.Init()
        sigFactory.GetSigSize()
        return Pharmacophore2DFeaturizer()

    if featurizer_choice == 'ErG':
        return ErGFeaturizer()

    if featurizer_choice == 'Morgan_rdkit':
        return [
            dc.feat.CircularFingerprint(size=1024),
            dc.feat.RDKitDescriptors()
        ]

    if featurizer_choice == 'AtomPairs_rdkit':
        return [AtomPairFeaturizer(), dc.feat.RDKitDescriptors()]

    if featurizer_choice in ['PharmacoPFP_rdkit', 'PharamacoPFP_rdkit']:
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        sigFactory = SigFactory(featFactory, minPointCount=2, maxPointCount=3)
        sigFactory.SetBins([(1, 3), (3, 8)])
        sigFactory.Init()
        sigFactory.GetSigSize()
        return [Pharmacophore2DFeaturizer(), dc.feat.RDKitDescriptors()]

    if featurizer_choice == 'MACCS_rdkit':
        return [dc.feat.MACCSKeysFingerprint(), dc.feat.RDKitDescriptors()]

    if featurizer_choice == 'FP2_rdkit':
        return [FP2fingerprint(), dc.feat.RDKitDescriptors()]

    if featurizer_choice in ['rdkit', 'Rdkit']:
        return dc.feat.RDKitDescriptors()

    if featurizer_choice in ['FPGNN', 'Chemprop']:
        return NoFeaturizer()


def load_model_ofa(name, algorithm, featurizer, best_hyperparams, seed1):
    tasks = ['standard_value']  # debug
    if algorithm in ['RF', 'SVM', 'XGB', 'KNN', 'NB']:
        model = pickle.load(
            open(
                os.path.join(
                    model_path,
                    "{}_{}_{}_{}.pkl".format(name, algorithm, featurizer,
                                             seed1)), "rb+"))

        return model

    if algorithm in ['DNN']:
        md_name = os.path.join(
            model_path, '{}_{}_{}_{}.h5'.format(name, algorithm, featurizer,
                                                seed1))
        model = tf.keras.models.load_model(md_name.encode('utf-8'))
        return model

    if algorithm == 'GCN':
        model = GraphConvModel(len(tasks),
                               batch_size=32,
                               mode='classification',
                               weight_decay=best_hyperparams[0],
                               graph_conv_layers=best_hyperparams[1],
                               learning_rate=best_hyperparams[2],
                               dense_layer_size=best_hyperparams[3])
        model_dir = os.path.join(model_path,
                                 '{}_{}_{}'.format(name, algorithm, seed1))
        model.restore(model_dir=model_dir)

        return model

    if algorithm == 'GAT':
        model = GATModel(mode='classification',
                         n_tasks=len(tasks),
                         batch_size=32,
                         weight_decay=best_hyperparams[0],
                         learning_rate=best_hyperparams[1],
                         n_attention_heads=best_hyperparams[2],
                         dropout=best_hyperparams[3])
        model.restore(model_dir=os.path.join(
            model_path, '{}_{}_{}'.format(name, algorithm, seed1)))
        return model

    if algorithm == 'MPNN':
        model = MPNNModel(mode='classification',
                          n_tasks=len(tasks),
                          batch_size=32,
                          weight_decay=best_hyperparams[0],
                          learning_rate=best_hyperparams[1],
                          graph_conv_layers=best_hyperparams[2],
                          num_layer_set2set=best_hyperparams[3],
                          node_out_feats=best_hyperparams[4],
                          edge_hidden_feats=best_hyperparams[5])
        model.restore(model_dir=os.path.join(
            model_path, '{}_{}_{}'.format(name, algorithm, seed1)))
        return model

    if algorithm == 'AttentiveFP':
        model = AttentiveFPModel(mode='classification',
                                 n_tasks=len(tasks),
                                 batch_size=32,
                                 dropout=best_hyperparams[0],
                                 graph_feat_size=best_hyperparams[1],
                                 num_timesteps=best_hyperparams[2],
                                 num_layers=best_hyperparams[3],
                                 learning_rate=best_hyperparams[4],
                                 weight_decay_penalty=best_hyperparams[5])
        model.restore(model_dir=os.path.join(
            model_path, '{}_{}_{}'.format(name, algorithm, seed1)))
        return model
    if algorithm == 'FPGNN':
        model = load_model(model_path + '/' + 'FPGNN' + '/' + name + '.pt',
                           torch.cuda.is_available())
        model.eval()
        return model
    if algorithm == 'Chemprop':
        model = load_checkpoint(model_path + '/' + 'Chemprop' + '/' + name +
                                '.pt')
        model.eval()
        return model


def predict_ofa(name,
                algorithm,
                featurizer,
                model,
                test_dataset=None,
                test_feature=None,
                seed2=0):
    if test_feature is None:
        test_feature = test_dataset.X
    if len(featurizer.split('_')) == 2:
        if featurizer.split('_')[1] == 'rdkit':
            if test_feature is None:
                test_feature = test_dataset.X
            index, X_max, X_min = pickle.load(
                open(
                    os.path.join(
                        model_path, '{}_{}_{}_{}_rdkit_fmm.pkl'.format(
                            name, algorithm, featurizer, seed2)), 'rb'))
            dataset_sel_rdkit = test_feature[:, -208:][:, index.astype(int)]
            denominator = np.where((X_max - X_min) > 0, (X_max - X_min),
                                   np.ones_like(X_max - X_min))
            test_feature = np.concatenate([
                test_feature[:, :-208],
                np.nan_to_num((dataset_sel_rdkit - X_min) / denominator)
            ],
                                          axis=1)
    if featurizer == 'rdkit':
        if test_feature is None:
            dataset_x = test_dataset
        else:
            dataset_x = dc.data.NumpyDataset(test_feature)
        pkl_load = pickle.load(
            open(
                os.path.join(
                    model_path,
                    '{}_{}_{}_{}_rdkit_fmm.pkl'.format(name, algorithm,
                                                       featurizer, seed2)),
                'rb'))
        index, X_max, X_min = pkl_load
        dataset_sel_X = dataset_x.X[:, index.astype(int)]
        denominator = np.where((X_max - X_min) > 0, (X_max - X_min),
                               np.ones_like(X_max - X_min))
        test_feature = np.nan_to_num((dataset_sel_X - X_min) / denominator)

    if algorithm in ['GAT', 'MPNN', 'AttentiveFP']:
        ys = model.predict(test_dataset)
        y_pre = ys[:, 1]
        return y_pre

    if algorithm in ['RF', 'XGB', 'SVM', 'KNN', 'NB']:
        ys = model.predict_proba(test_feature)
        y_pre = ys[:, 1]
        return y_pre

    if algorithm in ['DNN']:
        ys = model.predict(test_feature)
        y_pre = np.array(ys[0][:, :, 1]).squeeze()
        return y_pre
    if algorithm in ['GCN']:
        ys = model.predict(test_dataset)
        y_pre = np.array(ys[:, :, 1]).squeeze()
        return y_pre

    if algorithm == 'FPGNN':
        with torch.no_grad():
            y_pre = np.squeeze(model(test_feature).cpu().numpy())
        return y_pre


model_path = r'./models'

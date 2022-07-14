import multiprocessing
import pandas as pd
import numpy as np
from chemprop.train import predict
from chemprop.data import MoleculeDataLoader, MoleculeDatapoint, MoleculeDataset
from deepchem.data import NumpyDataset
from voting_fusion_use import get_predict_helper
from ofa_smi_eval import featurizer_ofa, load_model_ofa, predict_ofa
import json
import time
import sys
import warnings

warnings.filterwarnings('ignore')


def specific_predict(smiles, uniprot_id, method_name, seed=None, hyper=None):
    algorithm_name = method_name.split('_', 1)[0]
    algorithm_name = method_name.split(
        '_', 1)[1] if algorithm_name == 'Graph' else algorithm_name
    featurizer_name = method_name.split(
        '_',
        1)[1] if method_name != 'FPGNN' and method_name != 'Chemprop' else None
    featurizer = featurizer_ofa(featurizer_name) if featurizer_name else None
    model = load_model_ofa(uniprot_id, algorithm_name, featurizer_name, hyper,
                           seed)
    if method_name == 'FPGNN':
        predict_result = float(model([smiles])[0][0])
    elif method_name == 'Chemprop':
        predict_result = float(
            np.array(
                predict(
                    model,
                    MoleculeDataLoader(MoleculeDataset(
                        [MoleculeDatapoint(smiles=[smiles])]),
                                       batch_size=1,
                                       num_workers=0))).squeeze())
    else:
        if isinstance(featurizer, list):
            test_dataset = NumpyDataset(X=np.concatenate([
                NumpyDataset(X=featurizer[0].featurize([smiles])).X,
                NumpyDataset(X=featurizer[1].featurize([smiles])).X
            ],
                                                         axis=1))
        else:
            test_dataset = NumpyDataset(X=featurizer.featurize([smiles]))
        predict_result = float(
            predict_ofa(uniprot_id,
                        algorithm_name,
                        featurizer_name,
                        model,
                        test_dataset=test_dataset,
                        seed2=seed).flatten())
    return {uniprot_id: [method_name, predict_result]}


def kipp_predict(smiles: str, model_type: int):
    assert model_type in range(0, 2)
    pool = multiprocessing.Pool()
    result = {}
    if model_type == 0:
        voting_df = pd.read_csv('./data/voting_accor.csv',
                                encoding='utf-8',
                                low_memory=False)
        for target_name, seed in zip(voting_df['name'].tolist(),
                                     voting_df['seed'].tolist()):
            pool.apply_async(get_predict_helper,
                             args=(smiles, target_name, seed),
                             callback=lambda x: result.update(x),
                             error_callback=lambda x: print(x))
    else:
        info_df = pd.read_csv('./data/info_best.csv').set_index('uniprot_id')
        for index, row in info_df.iterrows():
            pool.apply_async(specific_predict,
                             args=(smiles, index, row[0], row[1],
                                   eval(row[2]) if row[2] else None),
                             callback=lambda x: result.update(x),
                             error_callback=lambda x: print(x))
    pool.close()
    pool.join()
    return result


if __name__ == '__main__':
    assert len(sys.argv) == 3
    res = kipp_predict(sys.argv[1], int(sys.argv[2]))
    with open(
            f'result_{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.json',
            'w') as f:
        json.dump(res, f)

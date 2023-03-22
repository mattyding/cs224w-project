"""Module to load odor data into PyG format."""

import torch 
import pandas as pd 
import pyrfume
import torch_geometric.utils as pyg_utils

def get_graph_data(name):
    """Load the specified dataset in PyG format.

    Supports the Leffingwell dataset ("lw") and the Dream Olfaction dataset ("ol").
    """
    return df_to_pyg(load_dataset_df(name))

def load_dataset_df(name):
    """Load the specified dataset into a DataFrame."""
    if name == 'lw': 
        # leffingwell dataset
        lw_behavior = pyrfume.load_data('leffingwell/behavior.csv', remote=True)

        pungent_lw = set(lw_behavior[lw_behavior['pungent'] == 1]['IsomericSMILES'])
        not_pungent_lw = set(lw_behavior[lw_behavior['pungent'] == 0]['IsomericSMILES'])
        df_lw = {chem: 1 if chem in pungent_lw else 0 for chem in pungent_lw.union(not_pungent_lw)}
        return pd.DataFrame(df_lw.items(), columns=['SMILES', 'pungent'])

    elif name == 'ol': 
        # dream olfaction dataset 
        df_ol = pd.read_csv('cs224w-project/data/ol_train.csv')

        pungent_ol = set(df_ol[df_ol['SENTENCE'].str.contains('pungent')]['SMILES'])
        not_pungent_ol = set(df_ol[~df_ol['SENTENCE'].str.contains('pungent')]['SMILES'])
        df_ol = {chem: 1 if chem in pungent_ol else 0 for chem in pungent_ol.union(not_pungent_ol)}
        return pd.DataFrame(df_ol.items(), columns=['SMILES', 'pungent'])

    else:
        print(f"Invalid dataset name specified: '{name}' must be 'ol' or 'lw'")
        raise ValueError

def df_to_pyg(df):
    """Convert a DataFrame of SMILES and labels to a list of PyG Data objects.
    
    Use existing node features and pool edge features to retrieve per-atom features.
    """
    graph_list = [pyg_utils.smiles.from_smiles(df['SMILES'][i]) for i in range(len(df))]

    for i in range(len(df)):
        graph_list[i].y = torch.Tensor([df['pungent'][i]])

    for i in range(len(df)):
        graph_list[i].x = graph_list[i].x.float()

    # incorporate edge features (sum pooling)
    for mol in graph_list:
        if (len(mol.edge_index) == 0) or len(mol.edge_attr) == 0:
            # no edge features, concat 0s
            mol.x = torch.cat((mol.x, torch.zeros((len(mol.x), 3))), dim=1)
            continue
        pooled_edge_features = []
        for atm_idx in range(len(mol.x)):
            sum_edge_feature = torch.zeros(len(mol.edge_attr[0]))
            for i, start_atm_idx in enumerate(mol.edge_index[0]):
                if start_atm_idx == atm_idx:
                    sum_edge_feature = torch.add(sum_edge_feature, mol.edge_attr[i])
            pooled_edge_features.append(sum_edge_feature)
        mol.x = torch.cat((mol.x, torch.stack(pooled_edge_features)), dim=1)

    return graph_list 
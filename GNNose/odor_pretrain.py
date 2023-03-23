"""Module for pretraining GNNs using contrastive learning"""

# use graphssl lib for GraphCL perturbations
from loss import infonce
from data import MyDataset
from torch_geometric.loader import DataLoader

def build_pretraining_loader(dataset, subset, augment_list=["edge_perturbation", "node_dropping"], batch_size=128):
    """Builds a dataloader by using the GraphSSL `MyDataset` class"""
    shuffle = (subset != "test")
    loader = DataLoader(
        MyDataset(dataset, subset, augment_list),
        batch_size=batch_size, 
        shuffle=shuffle, 
        follow_batch=["x_anchor", "x_pos"]
    )
    return loader

def pretrain(model, optimizer, epoch, mode, dataloader, device):
    """Run a single epoch of pretraining, adapted from GraphSSL"""
    if mode == "train":
        model.train()
    elif mode == "val" or mode == "test":
        model.eval()

    # contrastive_fn = jensen_shannon() # default to infonce loss 
    contrastive_fn = infonce() # default to infonce loss 

    total_loss = 0

    for data in dataloader:
        data.to(device)
        # readout_anchor is the embedding of the original datapoint x on passing through the model
        readout_anchor = model(data.x_anchor, 
                    data.edge_index_anchor, data.x_anchor_batch) # removed wrapping tup

        # readout_positive is the embedding of the positively augmented x on passing through the model
        readout_positive = model(data.x_pos, 
                    data.edge_index_pos, data.x_pos_batch) # removed wrapping tup

        # negative samples for calculating the contrastive loss is computed in contrastive_fn
        loss = contrastive_fn(readout_anchor, readout_positive)

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # keep track of loss values
        total_loss += loss.item() 

    # gather the results for the epoch
    epoch_loss = total_loss / len(dataloader.dataset)
    return epoch_loss
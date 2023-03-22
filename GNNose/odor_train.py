import torch 
import numpy as np

sigmoid_cutoff = 0.5 

def train(model, optimizer, train_loader, mode, device, weighted_BCE=False, ema=None):
    def get_class_weights():
        one_count = 0
        zero_count = 0
        for data in train_loader:
            one_count += torch.sum(data.y==1)
            zero_count += torch.sum(data.y==0)
        weight = torch.tensor([zero_count/(one_count+zero_count), one_count/(one_count+zero_count)])
        diff = (weight[0] - weight[1]) 
        return [1 - ((1 - diff) / 2), (1 - diff) / 2]

    if mode == "train":
        model.train()
    elif mode == "val" or mode == "test":
        model.eval()

    total_loss = 0
    total_graphs = 0
    
    weight = get_class_weights()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        
        if ema:
            with ema.average_parameters():
                if weighted_BCE:
                    batch_weight = (data.y==1) * weight[0] + (data.y==0)*weight[1]
                    loss = torch.nn.BCELoss(weight=batch_weight)(out.squeeze(), data.y)
                else:
                    loss = torch.nn.BCELoss()(out.squeeze(), data.y)
        else:
            if weighted_BCE:
                batch_weight = (data.y==1) * weight[0] + (data.y==0)*weight[1]
                loss = torch.nn.BCELoss(weight=batch_weight)(out.squeeze(), data.y)
            else:
                loss = torch.nn.BCELoss()(out.squeeze(), data.y)

        if mode == "train":
            loss.backward()
            optimizer.step()
            if ema:
                ema.update()

        total_loss += loss.item()

    return total_loss / len(train_loader.dataset)  # normalized by size of set

@torch.no_grad()
def test(model, loader, device):
    model.eval()

    total_correct = 0
    total_ex = 0 
    all_preds = []
    all_true = [] 
    for data in loader:
        data = data.to(device)

        pred = model(data.x, data.edge_index, data.batch)
        pred = (pred.squeeze() >= sigmoid_cutoff).float()

        total_correct += int((pred == data.y).sum()) 
        total_ex += np.prod(data.y.shape) 

        all_preds.append(pred)
        all_true.append(data.y)

    return total_correct / total_ex, torch.vstack(all_preds), torch.vstack(all_true)

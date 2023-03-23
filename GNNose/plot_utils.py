"""Module for plotting loss curves and visualizing molecules."""
import matplotlib.pyplot as plt 
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from data import MyDataset


def plot_losses(train_losses, val_losses, title=None):
    """Utility method to plot paired loss curves"""
    plt.plot(range(len(val_losses)), val_losses, label='val')
    plt.plot(range(len(train_losses)), train_losses, label='train')
    plt.legend()
    plt.title(title)
    plt.show()

def plot_loss(losses, title=None):
    """Utility method to plot single loss curve"""
    plt.plot(range(len(losses)), losses)
    plt.title(title)
    plt.show()

def nx_draw_molecule(x, edge_index, title=None, savepath=None):
    """Draw a molecule as a networkx graph"""
    data = Data(x=x, edge_index=edge_index)
    networkX_graph = to_networkx(data, node_attrs=["x"])
    nx.draw_networkx(
        networkX_graph, 
        pos=nx.shell_layout(networkX_graph),
        with_labels=False
    )
    if title:
        plt.title(title.replace('_', ' '))
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()

def plot_augmentation_example(val_set, example_index, augmentation_type, savedir='.'):
    """
    Visualize an example of an original vs. augmented graph from the dataset.

    Example usage:    
    plot_augmentation_example(val_set, 1, 'node_dropping')
    plot_augmentation_example(val_set, 1, 'edge_perturbation')
    """

    cur_dataset = MyDataset(val_set, "test", [augmentation_type])
    cur_ex = cur_dataset[example_index]

    nx_draw_molecule(
        cur_ex.x_anchor, cur_ex.edge_index_anchor, "original",
        savepath=f"{savedir}/original.png"
    )
    nx_draw_molecule(
        cur_ex.x_pos, cur_ex.edge_index_pos, augmentation_type,
        savepath=f"{savedir}/{augmentation_type}.png"
    )

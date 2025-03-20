import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor

class TQCeLoss(torch.nn.Module):
    def __init__(self, num_useful_qubits, num_meas_qubits):
        super().__init__()
        
        self.filter_matrix = np.zeros((num_meas_qubits, num_useful_qubits))
        for i in range(num_useful_qubits):
            self.filter_matrix[i, i] = 1
            
        self.filter_matrix = torch.from_numpy(self.filter_matrix).to(torch.float32)
            
    def forward(self, preds, labels):
        useful_preds = torch.matmul(preds, self.filter_matrix)
        loss = torch.nn.functional.cross_entropy(useful_preds, labels.long().squeeze())
            
        return loss

def auc_binary(preds, labels):
    y_pred_label=[]
    loss=torch.gt(torch.multiply(Tensor(preds), Tensor(labels)), 0)
    for i,j in zip(loss,labels):
        if i:
            y_pred_label.append(j)
        else:
            y_pred_label.append(j*(-1))
    return roc_auc_score(labels,y_pred_label)

def auc_multi_class(preds, labels, num_meas_qubits):
    softmax_outputs = torch.nn.functional.softmax(Tensor(preds[:, :num_meas_qubits]), dim=1)
    softmax_outputs_np = softmax_outputs.detach().numpy()
    auc_score = roc_auc_score(labels, softmax_outputs_np, multi_class='ovr')
    return auc_score

def auc_oct(preds,labels,num_meas_qubits):
    loss=torch.gt(torch.multiply(preds[:,:num_meas_qubits], torch.tensor(labels)), 0)
    pred_labels=[]
    for index in range(len(labels)):
        pred_label=[]
        for i, j in zip(loss[index], labels[index]):
            if i:
                pred_label.append(j)
            else:
                pred_label.append(j*(-1))
        pred_labels.append(pred_label)
    return roc_auc_score(labels, np.array(pred_labels),average=None)

def mse_batch_loss(preds, labels, mean=True):
    """
    MSE sample loss for a batch of predictions and labels. Use only with 1-dimensional predictions.
    """
    losses = np.power(np.subtract(preds, labels), 2)

    if mean:
        losses = np.mean(losses)
        
    return losses 


def mse_vec_batch_loss(preds, labels):
    """
    MSE loss for a batch of predictions and labels. Use only with 2-dimensional batch of predictions.
    """
    return np.mean(np.sum(np.power(np.subtract(preds, labels), 2), 1))


def batch_acc(preds, labels):
    """
    Accuracy for a batch of predictions and labels. Use only with 1-dimensional predictions.
    """    
    return np.mean(mse_batch_loss(preds, labels, False) < 1)


def vec_batch_acc(preds, labels):
    """
    Accuracy for a batch of predictions and labels. Use nly with 2-dimensional batch of predictions.
    """
    return np.mean(np.sum(np.multiply(preds, labels) > 0, 1) == preds.shape[1])
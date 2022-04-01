import os
import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_accuracy(logits, labels):

    # find location of the max value of the logits
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(device)
    # make it so the index with the best answer has a value of 1
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    # take the sum of each row to count the overlap with analyst answers
    row_sums = torch.sum(scores, 1)
    row_sums_div_3 = torch.div(row_sums, 3.0)
    ones = torch.ones(row_sums.size()).to(device)
    row_sums_ones = torch.cat((row_sums_div_3.unsqueeze(-1), ones.unsqueeze(-1)),dim=1)
    accuracies = torch.min(row_sums_ones, dim=1)
    print(accuracies.values)
    return accuracies.values


def test_model(val_dataloader, model):
    model.eval()
    torch.set_grad_enabled(False)
    upper_bound = 0
    num_data = 0
    total_loss = 0
    accuracy = torch.tensor([]).to(device)

    for batch_quest, batch_ans, batch_vis in val_dataloader:
        batch_quest, batch_ans, batch_vis = batch_quest.to(device) , batch_ans.to(device).float() , batch_vis.to(device)
        logits = model(batch_vis, batch_quest)
        loss = instance_bce_with_logits(logits, batch_ans).to(device)
        curr_acc = compute_accuracy(logits, batch_ans)
        accuracy = torch.cat((accuracy, curr_acc), dim=0)
        upper_bound += (batch_ans.max(1)[0]).sum()
        num_data += logits.size(0)
        total_loss += loss.item() * batch_vis.size(0)

    print(accuracy.sum())
    print(torch.mean(accuracy).item())
    x = input('yo')
    val_loss = total_loss/len(val_dataloader)
    return val_loss, torch.mean(accuracy).item()

def train_model(train_dataloader, val_dataloader, model, n_epochs, save_loc, device):

    optimizer = torch.optim.Adam(list(model.parameters()), lr=5e-5)
    history = dict(train=[], val=[], val_acc=[])
    best_loss = 10000.0
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        total_loss = 0
        print('Starting Epoch: {}'.format(epoch))
        model.train()
        torch.set_grad_enabled(True)
        for batch_quest, batch_ans, batch_vis in train_dataloader:
            batch_quest, batch_ans, batch_vis = batch_quest.to(device) , batch_ans.to(device).float() , batch_vis.to(device)
            optimizer.zero_grad() # zero the parameter gradients
            logits = model(batch_vis, batch_quest)#, batch_ans)
            #loss = instance_bce_with_logits(logits, batch_ans).to(device)
            loss = criterion(logits, batch_ans).to(device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_vis.size(0)
        epoch_loss = total_loss/len(train_dataloader)
        print("Training Loss: {0} - Epoch: {1}".format(round(epoch_loss,8), epoch+1))

        val_loss, val_acc = test_model(val_dataloader, model)
        print(val_acc)

        history['val'].append(val_loss)

        history['val_acc'].append(val_acc)
        history['train'].append(epoch_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model, save_loc+'/best_model.pth')
    print(history)
    return history



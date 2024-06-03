'''Tests for train_bir.py'''
import os
import torch
from src.train_bir import load_data, setup_model, save_model
from src.model_bir import BirNet

def test_load_data():
    trainloader, valloader, testloader = load_data()
    assert isinstance(trainloader, torch.utils.data.DataLoader)
    assert isinstance(valloader, torch.utils.data.DataLoader)
    assert isinstance(testloader, torch.utils.data.DataLoader)

def test_setup_model():
    net, criterion, optimizer = setup_model()
    assert isinstance(net, BirNet)
    assert isinstance(criterion, torch.nn.MSELoss)
    assert isinstance(optimizer, torch.optim.Adam)

def test_save_model(tmpdir):
    net, _, _ = setup_model()
    save_dir = tmpdir.mkdir("subdir")
    save_model(net, str(save_dir))
    assert os.path.exists(save_dir)

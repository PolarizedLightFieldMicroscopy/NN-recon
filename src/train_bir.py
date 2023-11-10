'''Training script to train the model defined in model_bir.py
Parameters and progress can be visualized with the command: 
    tensorboard --logdir=runs
'''
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from src.model_bir import BirNetwork, BirNetworkDense, BirNet
from src.Data import BirefringenceDataset

DATA_PATH = "../../NN_data/small_sphere_random_bir1000/spheres_11by11"
RUN_NAME = 'conv3d_after_fully_connected'
SAVE_DIR = "../../../NN_data/small_sphere_random_bir1000/models/BirNet_Nov10/" + RUN_NAME + '/'
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"using {DEVICE} device")

def load_data():
    '''This function loads the data from the specified path and returns
    the train, validation, and test data loaders.'''
    print(f'loading data from {DATA_PATH}')
    train_data = BirefringenceDataset(DATA_PATH, split='train', source_norm=True, target_norm=True)
    val_data = BirefringenceDataset(DATA_PATH, split='val', source_norm=True, target_norm=True)
    test_data = BirefringenceDataset(DATA_PATH, split='test', source_norm=True, target_norm=True)
    batch_size = 1
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    return trainloader, valloader, testloader

def setup_model():
    '''Instantiates the model, loss function, and optimizer.'''
    net = BirNet(target_conv=True).to(DEVICE)
    print(summary(net, (512, 16, 16)))
    print("model instantiated")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    return net, criterion, optimizer

def train_one_epoch(net, trainloader, loss_function, optimizer, tfwriter, epoch):
    '''This function updates the weights of the network as it loops
    through all the data in the dataset.'''
    running_loss = 0.0
    running_loss_per_epoch = 0.0
    for i, data in enumerate(trainloader, start=0):
        net.train()
        source, target = data
        source = source.to(DEVICE)
        target = target.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(source)
        train_loss = loss_function(outputs, target)
        train_loss.backward()
        optimizer.step()

        # print statistics
        running_loss += train_loss.item()
        running_loss_per_epoch += train_loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            tfwriter.add_scalar('Loss/train', running_loss / 50, i)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
            running_loss = 0.0

        data_idx = i
    avg_sum_per_epoch = running_loss_per_epoch / data_idx
    return net, avg_sum_per_epoch

def validate(net, valloader, loss_function, optimizer, tfwriter, epoch):
    '''This function validates network parameter optimizations'''
    running_loss = 0.0
    val_loss_per_batch = []
    net.eval()
    print('validating...')
    #  iterating through batches
    for i, data in enumerate(valloader, start=0):
        source, target = data
        source = source.to(DEVICE)
        target = target.to(DEVICE)
        # zero the parameter gradients
        optimizer.zero_grad()
        output = net(source)
        val_loss = loss_function(output, target)
        val_loss_per_batch.append(val_loss.item())
        # print statistics
        running_loss += val_loss.item()
        if i % 50 == 49:    # print every 50 mini-batches
            tfwriter.add_scalar('Loss/eval', running_loss / 50, i)
            print(f'[{epoch + 1}, {i + 1:5d}] val loss: {running_loss:.3f}')
            running_loss = 0.0
    print('all done!')
    return val_loss_per_batch

def save_model(trained_net, save_dir, epoch=None):
    '''Saves the model to the specified directory'''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_filename = save_dir + (f'epoch{epoch}.pt' if epoch is not None else 'final.pt')
    torch.save(trained_net.state_dict(), model_filename)
    print(f'saved model as {model_filename}')

def train_model(net, trainloader, valloader, criterion, optimizer):
    '''Trains the model for 100 epochs and saves the model'''
    writer = SummaryWriter('runs/' + RUN_NAME)
    min_val_loss = 1000
    for epoch in range(100):  # loop over the dataset multiple times
        print(f"starting training epoch {epoch}")
        trained_net, train_loss_per_batch = train_one_epoch(
            net, trainloader, criterion, optimizer, writer, epoch
        )
        writer.add_scalar('Loss/train per epoch', train_loss_per_batch, epoch)

        # validate after each epoch
        val_loss_per_batch = validate(trained_net, valloader, criterion, optimizer, writer, epoch)
        writer.add_scalar('Loss/validate per epoch', np.mean(val_loss_per_batch), epoch)

        if np.mean(val_loss_per_batch) < min_val_loss:
            save_model(trained_net, SAVE_DIR, epoch)
            min_val_loss = np.mean(val_loss_per_batch)
        print(f'mean val loss: {np.mean(val_loss_per_batch):.6f}, '
              f'current min val loss: {min_val_loss:.6f}'
        )


if __name__ == '__main__':
    data_trainloader, data_valloader, data_testloader = load_data()
    network, loss_criterion, loss_optimizer = setup_model()
    train_model(network, data_trainloader, data_valloader, loss_criterion, loss_optimizer)
    print('Finished Training')

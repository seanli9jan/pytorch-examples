import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import OrderedDict
import io

# TenaorBoard
from torch.utils.tensorboard import SummaryWriter

# Ignore warnings
#import warnings
#warnings.filterwarnings("ignore")

# flags and parameters
network = 'LeNet'
no_cuda = False

batch_size = 64
test_batch_size = 1000
epochs = 10
learning_rate = 1e-3
use_custom_loss = True
n_iter = 0

random_seed = False
log_interval = 10

save_model = True
load_model = False
parameters_only = False

change_layer = True
keep_layers_name = True
model_log = True

if network == 'LeNet':
    from lenet import Net
elif network == 'UNet':
    from unet import Net

# Writer will output to ./runs/ directory by default
writer = SummaryWriter(network)

# loss function
def custom_loss(output, target, device):
    target = torch.zeros(len(target), 10).to(device).scatter_(1, target.reshape(len(target), 1), 1)
    return torch.mean(torch.sum(-(target * output), 1))

# training rule
def train(model, device, train_loader, optimizer, epoch):
    global n_iter
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if use_custom_loss:
            loss = custom_loss(output, target, device).requires_grad_(True)
        else:
            loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            n_iter += log_interval
            writer.add_scalar('Loss/train', loss.item(), n_iter)

# testing rule
def test(model, device, test_loader):
    global n_iter
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if use_custom_loss:
                test_loss += custom_loss(output, target, device).item()
            else:
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    writer.add_scalar('Loss/test', test_loss, n_iter)

# train/test set
def load_data(use_cuda):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader

def main():
    # gpu setting
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # random seed
    if random_seed:
        torch.manual_seed(1)

    # load data
    train_loader, test_loader = load_data(use_cuda)

    # define network
    model = Net().to(device)

    # load model
    if load_model:
        with open(network+'_model.pt', 'rb') as f:
            graph = f.read()

        if parameters_only:
            model.load_state_dict(torch.load(io.BytesIO(graph))).to(device)
            # without open()
            #model.load_state_dict(torch.load(network+'_model.pt')).to(device)
        else:
            model = torch.load(io.BytesIO(graph)).to(device)
            # without open()
            #model = torch.load(network+'_model.pt').to(device)

    if change_layer:
        if network == 'LeNet':
            # remove layer
            if keep_layers_name:
                # keep layers name
                model.fc.fc = nn.Sequential(OrderedDict([
                    *list(model.fc.fc.named_children())[:-2]
                ]))
            else:
                # default layers name
                model.fc.fc = nn.Sequential(
                    *list(model.fc.fc.children())[:-2]
                )

            # frozen
            for param in model.parameters():
                param.requires_grad = False

            # add new layer
            if keep_layers_name:
                # keep layers name
                """
                nn.Sequential(OrderedDict([
                    ...
                    ('layer_name1', nn.layer())
                    ('layer_name2', nn.layer())
                    ...
                ]))
                """
                model.fc.fc = nn.Sequential(OrderedDict([
                    *list(model.fc.fc.named_children()),
                    (str(len(model.fc.fc)+0), nn.ReLU(inplace=True)),
                    (str(len(model.fc.fc)+1), nn.Linear(500, 10))
                ])).to(device)
            else:
                # default layers name
                model.fc.fc = nn.Sequential(
                    *list(model.fc.fc.children()),
                    nn.ReLU(inplace=True),
                    nn.Linear(500, 10)
                ).to(device)
        elif network == 'UNet':
            # remove layer
            if keep_layers_name:
                # keep layers name
                model.up4.conv.conv = nn.Sequential(OrderedDict([
                    *list(model.up4.conv.conv.named_children())[:-3]
                ]))
            else:
                # default layers name
                model.fc.fc = nn.Sequential(
                    *list(model.up4.conv.conv.children())[:-3]
                )

            # frozen
            for param in model.parameters():
                param.requires_grad = False

            # add new layer
            if keep_layers_name:
                # keep layers name
                """
                nn.Sequential(OrderedDict([
                    ...
                    ('layer_name1', nn.layer())
                    ('layer_name2', nn.layer())
                    ...
                ]))
                """
                model.up4.conv.conv = nn.Sequential(OrderedDict([
                    *list(model.up4.conv.conv.named_children()),
                    (str(len(model.up4.conv.conv)+0), nn.Conv2d(64, 64, 3, padding=1)),
                    (str(len(model.up4.conv.conv)+1), nn.BatchNorm2d(64)),
                    (str(len(model.up4.conv.conv)+2), nn.ReLU(inplace=True))
                ])).to(device)
            else:
                # default layers name
                model.up4.conv.conv = nn.Sequential(
                    *list(model.up4.conv.conv.children()),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                ).to(device)

    if model_log:
        """
        model
        model.layer_name.weight
        model.layer_name.bias
        *model.parameters()
        """
        print(model)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer.add_graph(model, torch.rand(1, 1, 28, 28).to(device))

    # start training/testing
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # save model
    if (save_model):
        if parameters_only:
            torch.save(model.state_dict(), network+'_model.pt')
        else:
            torch.save(model, network+'_model.pt')

    writer.close()

if __name__ == '__main__':
    main()

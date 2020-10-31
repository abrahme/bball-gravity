from utils.data_loader import GravityDataSet, ToTensor
from torchvision import transforms
from torch.utils.data import DataLoader
from model.Tensor import *
import torch.optim as optim

if __name__ == "__main__":
    tl.set_backend("pytorch")
    batch_size = 16
    device = "cpu"

    dataset = GravityDataSet(T=960, P=24, Q=47, R=50, max_x=47, max_y=50,
                             data_dir_globbed="processed_data/possession_data/*.pkl",
                             player_encoding_path="processed_data/player_map.json",
                             transform=transforms.Compose([ToTensor()]))

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TRL(input_size=(batch_size, 960, 48, 47, 50),
                ranks=(10,3,3,10), output_size=(batch_size, 1))

    optimizer = optim.SGD(model.parameters(), lr=.1, momentum=.9)
    criterion = nn.MSELoss()

    n_epochs = 1
    regularizer = .01
    model = model.to(device)

    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            x = data.to(device)
            y = target.to(device)
            optimizer.zero_grad()
            output = model(x.float())
            loss = criterion(x, y) + regularizer * model.penalty(2)
            loss.backward()
            optimizer.step()
            if batch_idx % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss))

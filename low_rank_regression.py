from utils.data_loader import GravityDataSet, ToTensor
from utils.data_utils import read_json
from torchvision import transforms
from torch.utils.data import DataLoader
from model.Tensor import *
import torch.optim as optim

if __name__ == "__main__":
    tl.set_backend("pytorch")
    batch_size = 8
    device = "cpu"

    player_map = read_json("processed_data/player_map.json")
    num_players = len(player_map)
    x_res = 24
    y_res = 25

    dataset = GravityDataSet(T=960, P=num_players, Q=x_res, R=y_res, max_x=47, max_y=50,
                             data_dir_globbed="processed_data/possession_data/*.pkl",
                             player_encoding_map=player_map,
                             transform=transforms.Compose([ToTensor()]))

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CPRL(input_size=(batch_size, 960, 2 * num_players, x_res, y_res),
                 rank=2, output_size=(batch_size, 1))

    optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.9)
    criterion = nn.MSELoss()
    n_epochs = 1
    regularizer = .01
    model = model.to(device)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            x = data.to(device)
            y = target.to(device)
            optimizer.zero_grad()
            output = model(x.float())
            mse_loss = criterion(output, y.float())
            loss = mse_loss + regularizer * model.penalty(2)
            epoch_loss += mse_loss * x.shape[0]
            loss.backward()
            optimizer.step()
            if batch_idx % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss))
        print(f"Epoch: {epoch}, Loss: {epoch_loss / len(dataset)}")

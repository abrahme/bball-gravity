from utils.data_loader import GravityDataSet, ToTensor
from utils.data_utils import read_json
from torchvision import transforms
from torch.utils.data import DataLoader
from model.Tensor import *
from model.model_utils import save_model
import torch.optim as optim
import argparse
import yaml


def read_args(arg_path: str):
    """

    :param arg_path: where to get the arguments from
    :return:
    """
    with open(arg_path, "r") as f:
        arg_yaml = yaml.load(f)
    f.close()
    parser = argparse.ArgumentParser(description='Train gravity tensor model')
    for arg in arg_yaml:
        name = f"--{arg['name']}"
        dest = arg["dest"]
        help_desc = arg["help"]
        default_val = arg["default"]
        action = arg["action"]
        parser.add_argument(name, dest=dest, action=action, default=default_val, help=help_desc, required=False)
    return parser.parse_args()


if __name__ == "__main__":

    parsed_args = read_args("configs/args.yaml")
    player_file = parsed_args.player_data_file
    backend = parsed_args.backend
    tl.set_backend(backend)
    player_map = read_json(player_file)
    num_players = len(player_map)
    x_res = parsed_args.Q
    y_res = parsed_args.R
    batch_size = parsed_args.batch_size
    device = parsed_args.device
    lr = parsed_args.learning_rate

    dataset = GravityDataSet(T=960, P=num_players, Q=x_res, R=y_res, max_x=47, max_y=50,
                             data_dir_globbed="processed_data/possession_data/*.pkl",
                             player_encoding_map=player_map,
                             transform=transforms.Compose([ToTensor()]))

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CPRL(input_size=(batch_size, 960, 2 * num_players, x_res, y_res),
                 rank=1, output_size=(batch_size, 1))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.9)
    criterion = nn.MSELoss()
    n_epochs = parsed_args.n_epochs
    regularizer = parsed_args.regularizer
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

    save_model(model,parsed_args.save_path)


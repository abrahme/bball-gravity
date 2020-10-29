from utils.data_loader import GravityDataSet, ToTensor
from torchvision import transforms

if __name__ == "__main__":
    dataset = GravityDataSet(T=960, P=19, Q=47, R=50, max_x=47, max_y=50,
                             data_dir_globbed="processed_data/possession_data/*.pkl",
                             player_encoding_path="processed_data/player_map.json",
                             transform=transforms.Compose([ToTensor()]))



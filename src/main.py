import os
import tarfile
import wget
from pycocotools.coco import COCO
import requests

from torch.utils.data import Dataset, DataLoader

def download_dataset(path='./input/coco_person'):
    if not os.path.exists(path):
        os.mkdir(path)
        # instantiate COCO specifying the annotations json path
        coco = COCO('./input/annotations/person_keypoints_train2017.json')
        # Specify a list of category names of interest
        catIds = coco.getCatIds(catNms=['person'])
        # Get the corresponding image ids and images using loadImgs
        imgIds = coco.getImgIds(catIds=catIds)
        images = coco.loadImgs(imgIds)

        # Save the images into a local folder
        for im in images:
            img_data = requests.get(im['coco_url']).content
            with open(os.path.join(path, im['file_name']), 'wb') as handler:
                handler.write(img_data)

# Dataset helper function
def read_image(path):
    return None

##################Data Loader #######3
class CUB(Dataset):
    def __init__(self, files_path, split, train=True):
        return
    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        return

def initialize_loader(train_batch_size=64, val_batch_size=64):
    return

def visualize_dataset(dataloader):
    return

def plot_prediction(args, model, is_train, index_list=[0], plotpath=None, title=None):
    return

##############helper function for training##############
def compute_loss(pred, gt):
    return

def run_validation_step(args, epoch, model, loader, plotpath=None):
    return

def train(args, model):
    return

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == '__main__':
    download_dataset()

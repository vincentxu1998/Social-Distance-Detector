import logging

logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

import os
import tarfile
import wget
from pycocotools.coco import COCO
import requests

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import src.transforms as T


def download_dataset(path='./input/coco_person'):
    if not os.path.exists(path):
        os.mkdir(path)
        # instantiate COCO specifying the annotations json path
        # Specify a list of category names of interest
        coco = COCO('./input/annotations/person_keypoints_train2017.json')
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
from PIL import Image
class CUB(Dataset):
    def __init__(self, files_path, transforms):
        self.files_path = files_path
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(files_path, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(files_path, "PedMasks"))))
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.files_path, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.files_path, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

def initialize_loader(train_batch_size=4, val_batch_size=4):
    train_dataset = CUB(PennFudanPath, get_transform(train=True))
    valid_dataset = CUB(PennFudanPath, get_transform(train=False))

    indices = torch.randperm(len(train_dataset)).tolist()
    dataset = torch.utils.data.Subset(train_dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(valid_dataset, indices[-50:])

    return

def visualize_dataset(dataloader):
    return
    # x, y = next(iter(dataloader))
    # cats = coco.loadCats(coco.getCatIds())
    # nms = [cat['name'] for cat in cats]
    # logger.info('COCO categories: \n{}\n'.format(' '.join(nms)))
    #
    # nms = set([cat['supercategory'] for cat in cats])
    # logger('COCO supercategories: \n{}'.format(' '.join(nms)))
    # fig = plt.figure(figsize=(10, 5))
    # for i in range(4):
    #     inp = x[i]
    #     inp = inp.numpy().transpose(1, 2, 0)
    #     inp = denormalize(inp)
    #     mask = y[i] / 255.
    #
    #     ax = fig.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
    #     plt.imshow(np.concatenate([inp, mask], axis=1))
    #     plt.show()

def plot_prediction(args, model, is_train, index_list=[0], plotpath=None, title=None):
    return

##############helper function for training##############
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

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
    PennFudanPath = r'./input/PennFudanPed'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    initialize_loader()





import os
import tarfile
import wget

from pycocotools.coco import COCO
import requests

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


if __name__ == '__main__':
    download_dataset()

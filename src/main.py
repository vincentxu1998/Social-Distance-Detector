
if __name__ == '__main__':

    import os
    import tarfile
    import wget
    if not os.path.exists("17flowers.tgz"):
        print("Downloading flower dataset")
        wget.download(r'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz')
        my_tar = tarfile.open('17flowers.tgz')
        my_tar.extractall('.')
        my_tar.close()
    if not os.path.exists("trimaps.tgz"):
        wget.download(r'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/trimaps.tgz')
        my_tar = tarfile.open('trimaps.tgz')
        my_tar.extractall('.')
        my_tar.close()
    if not os.path.exists("datasplits.mat"):
        wget.download(r'https://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat')

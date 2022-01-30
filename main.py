import os
import gdown
import logging

logging.basicConfig(level=logging.NOTSET)


def setup():
    print("Download frontal-face trained classifiers:")
    haarcascade_frontalface_default_url = "https://raw.githubusercontent.com/iradbouzidi/Gender-Recognition-Web-App/master/haarcascade_frontalface_default.xml"
    haarcascade_frontalface_default_xml = "haarcascade_frontalface_default.xml"
    os.system(f"curl {haarcascade_frontalface_default_url} --output {haarcascade_frontalface_default_xml}")
    print("Download Dataset:")
    dataset_url = "https://drive.google.com/uc?id=1WEso3i11j4BkZoQNLIdJSB_t2dnXwc8M"
    dataset_tar = "Dataset.tar.bz2"
    gdown.download(dataset_url, dataset_tar, quiet=False)
    os.system(f"tar -xjvf {dataset_tar} Dataset/")
    os.system(f"tar -xjvf {dataset_tar} Dataset/")


logging.info(" Setup Start.")
setup()
logging.info(" Setup Done.")
















#print("Full dataset found under https://www.kaggle.com/cashutosh/gender-classification-dataset")
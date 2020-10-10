import tensorflow as tf
import os
import sys
import random
import numpy as np
import cv2
from PIL import Image


def load_and_resize_img(img, size):
    h, w = size
    img = Image.open(img)
    img = img.convert('RGB')
    img = np.array(img)
    x, y, _ = img.shape
    if h is not None and w is not None:
        img = cv2.resize(img, (w, h))
    return img


class CelebaAttrDict:
    def __init__(self, anno_path):
        with open(anno_path, "r") as fd:
            content = fd.readlines()
        self.attribute_names = self._process_line(content[1])
        self.d = self._process(content[2:])

        self.domains = None

    def _process_line(self, l):
        return list(filter(lambda x: x not in [''], l.rstrip('\n').split(' ')))

    def _process(self, content):
        d = {}
        for l in content:
            l = self._process_line(l)
            file_id, attribute_values = l[0].split('.')[0], l[1:]
            attr_dict = {}
            for i in range(len(self.attribute_names)):
                name, value = self.attribute_names[i],  int(attribute_values[i]) == 1
                attr_dict[name] = value
            d[file_id] = attr_dict
        return d

    def split_by_domain(self, domains):
        self.domains = [{} for _ in domains]
        for k, v in self.d.items():
            for i in range(len(domains)):
                domain_attr = domains[i]
                belong = domain_attr(v)
                if belong:
                    self.domains[i][k] = v
                    # found an domain to which the img belongs
                    break

    def find(self, file_id):
        return self.d[file_id]

    def sample_by_domain(self, batch_size):
        sampled = []
        for d in self.domains:
            sampled_keys = random.sample(d.keys(),  batch_size)
            sampled.append(sampled_keys)
        return sampled


def save_batch(batch, prefix):
    for i in range(batch.shape[0]):
        img = batch[i]
        img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        pil_img = Image.fromarray(img.astype('uint8'))
        pil_img.save(os.path.join('out', "%s_%d.jpg" % (prefix, i)))



def sample_celeba_with_attributes(data_dir, anno_dir, batch_size, size, attribute_domains):
    """
    Returns:
        a list of np.ndarray. [img_batch_domain1, img_batch_domain2, ...]
    """
    attr_dict = CelebaAttrDict(anno_dir)
    attr_dict.split_by_domain(attribute_domains)
    def process(idx, size):
        filename = os.path.join(data_dir, idx + '.jpg')
        img = load_and_resize_img(filename, size)
        if random.random() > 0.5: img = cv2.flip(img, 1)
        return img

    while True:
        img_indices = attr_dict.sample_by_domain(batch_size)
        imgs = [np.array([process(i, size) for i in d]) for d in img_indices]
        yield imgs

def celeba_img_gen(batch_size):
    return sample_celeba_with_attributes(
        os.path.join('data', 'img_align_celeba_png'), # download the celeba dataset and specify the data dir here
        os.path.join('data', 'img_align_celeba_anno',  'list_attr_celeba.txt'),  # download the celeba dataset and specify the annotation dir here
        batch_size, 
        (192, 160), # resize the image into a target shape
        [
            lambda attr: attr['Male'], # lambda function to filter domain: male
            lambda attr: not attr['Male'] # lambda function to filter domain: female
        ]
    )



import matplotlib.pyplot as plt
def viz_batch(batch):
    n_imgs = batch.shape[0]
    plt.figure(figsize=(20, 40))
    for i in range(n_imgs):
        plt.subplot(1, n_imgs, i+1)
        plt.imshow(batch[i])

"""
NOTE: you may want to try
1. the usual logistic losses
2. WGAN loss and WGAN-GP loss
3. non-saturating logistic losses
"""
def gen_loss(tensor):
    # LSGAN Loss
    return tf.reduce_mean(tf.losses.mse(tf.zeros(shape=(4, 1)), tensor))

def recon_loss(a, b):
    # reconstruction loss
    # pixel-wise l2 loss
    return tf.reduce_mean(tf.losses.mse(a, b))

def dis_loss(r, f):
    # LSGAN Loss
    return tf.reduce_mean(
            tf.losses.mse(tf.zeros(shape=(4, 1)), r) + tf.losses.mse(tf.ones(shape=(4, 1)), f))
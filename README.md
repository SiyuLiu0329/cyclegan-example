# Description
Simple CycleGAN Demo Code. This is by no means a state of the art network as of 2020, so do not expect the results to be mind blowing (for that you may want to look into StarGANv2). 

To run the code you need to download the CelebA dataset and locate the following:
- `Img/img_align_celeba_png` directory (extracted from a compressed file
- `Anno/'list_attr_celeba.txt`
You need to specify the paths to the above in `celeba_img_gen` function defined in `misc.py`

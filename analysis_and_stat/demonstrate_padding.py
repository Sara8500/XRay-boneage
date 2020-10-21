import utilities
import os
from PIL import Image
import torchvision.transforms as transforms
import argparse


def parse_arguments():
    print('\nParse arguments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dataset', type=str, default=None,
                        help='path to training the dataset')
    parser.add_argument('--out_dir', type=str, default="./statistics_dataset",
                        help='path to output directory')
    arguments = parser.parse_args()
    return arguments


args = parse_arguments()

utilities.check_output_dir(args.out_dir)

all_images = utilities.get_list_of_files(args.training_dataset)

im_path = os.path.join(args.training_dataset,all_images[0])

img = Image.open(im_path)

w, h = img.size
max_dim = max([w, h])

w_pad = max_dim - w
h_pad = max_dim - h

print(w_pad)
print(h_pad)

tr = transforms.Pad((0, 0, w_pad, h_pad), padding_mode="edge")
tr_2 = transforms.Pad((0, 0, w_pad, h_pad), padding_mode='constant')

img2 = tr(img)
img3 = tr_2(img)

img.show()
img2.show()
img3.show()



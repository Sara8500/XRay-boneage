import os
import torch.utils.data as data
import utilities
import numpy as np
from PIL import Image
import math
import pandas as pd
import torchvision.transforms as transforms
import preprocess
import preprocess.kaggle_download as kaggle_download
import preprocess.create_annotation_files as create_annotation_files



import preprocess.crop_images as crop_images

__all__ = ['XRAYBONES']

"""
Dataset: 
Publication: 
"""


class XRAYBONES(data.Dataset):

    def __init__(self, root, dir_cropping_info=None, lr_ann_file_path=None, train=True, split=0.8, Nmax=None, task='age', transform=None):
        '''
         kaggle testing dataset does not have any annotations and can't be used to calculate accuracy.
         kaggle training dataset has to be split into training and testing dataset.
         e.g. if train = true and split = 0.8, the first 80 % of the sorted cropping files are used as a training dataset.
        '''

        self.__download_dataset(root)

        # dataset with cropped images: preprocess dataset
        if dir_cropping_info:
            self.crop = True
            self.__prepare_preprocessed_dataset(root, dir_cropping_info, train, split)

        # dataset with original images
        else:
            self.crop = False
            self.__prepare_annotations_working_with_original_dataset(root, lr_ann_file_path, train, split, Nmax)

        self.train = train
        self.task = task
        self.transform = transform

        if task == 'gender':
            self.category_list = ['Male', 'Female']
        elif task == 'leftorright':
            self.flag_flip_image = np.random.randint(0, 2, self.number_of_images)
            self.category_list = ['Left', 'Right']

    def __getitem__(self, index):
        """ getitem from dataset according to given index"""

        # take images from the cropped dataset:
        if self.crop:
            file_name = self.file_list[index]

            img = Image.open(os.path.join(self.dir_cropped_images, file_name)) #img is PIL image

            target = self.flag_flip_image[index]

            if self.task == "leftorright":

                if target == 1: # image should be right but is left
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)

            transformed_img = self.transform(img)  # the transform needs to resize all images to same size,
            # even when corrupting aspect ratio

            return transformed_img, target  # tuple

        # take images from the original dataset according to the annotations loaded in __init__() :
        else:

            # look at the line given by "index" in self.annotations (saved at the end of __init__())
            ann = self.annotations[index]

            img_path = os.path.join(self.dir_images, (str(ann["id"])+".png"))

            # open corresponding image
            img = Image.open(img_path)  # img is PIL image

            if self.task == "leftorright":
                target = self.flag_flip_image[index]
                if target == 1:  # image should be right but is left
                    if self.annotations[index]['is_left']:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                else:  # image should be left but is right
                    if self.annotations[index]['is_left'] == False:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif self.task == "gender":
                if ann["is_male"]:
                    target = 0
                else:
                    target = 1
            if self.task == "age":
                target = self.annotations[index]['boneage']

            w, h = img.size
            max_dim = max([w, h])

            w_pad = max_dim - w
            h_pad = max_dim - h

            transform_padding = transforms.Pad((0, 0, w_pad, h_pad), fill=0, padding_mode="constant")
            # padding: (left, top, right, bottom), padding_mode=edge

            img = transform_padding(img)

            # apply given transform (at least composed of resize all images to same size and toTensor):
            transformed_img = self.transform(img)

            return transformed_img, target  # tuple

    def __len__(self):
        return self.number_of_images

    def __repr__(self):
        raise NotImplementedError

    def __download_dataset(self, root):
        # download the dataset (if not already downloaded)

        self.dir_downloaded_dataset = os.path.join(root, "downloaded_dataset")

        if os.path.isdir(self.dir_downloaded_dataset):
            print('Dataset already downloaded. Delete "downloaded_dataset" folder to download dataset again.')

        else:
            print("downloading rsna-bone-age dataset from Kaggle. Please be patient. Download size 9GB...")
            os.makedirs(self.dir_downloaded_dataset)
            kaggle_download.download_dataset_kaggleAPI("kmader/rsna-bone-age", self.dir_downloaded_dataset)
            print("finished downloading, unzipping files...")
            kaggle_download.unzip_all_zipfiles(self.dir_downloaded_dataset)
            print("finished unzipping.")

    def __prepare_preprocessed_dataset(self, root, dir_cropping_info, train, split):
        # create pre-processed dataset with cropped images and annotation files (if necessary):

        self.dir_preprocessed_dataset = os.path.join(root, "preprocessed_dataset")

        if train:
            print("Preparing Train dataset:")
            self.dir_annotations = os.path.join(self.dir_preprocessed_dataset, "train_annoations")
            self.dir_cropped_images = os.path.join(self.dir_preprocessed_dataset, "train_images")
        else:  # test dataset
            print("Preparing Test dataset:")
            self.dir_annotations = os.path.join(self.dir_preprocessed_dataset, "test_annoations")
            self.dir_cropped_images = os.path.join(self.dir_preprocessed_dataset, "test_images")

        if os.path.isdir(self.dir_annotations):
            print('Preprocessed dataset already created. Delete "{}" to preprocess dataset again.'.format(
                self.dir_annotations))

        else:
            print("Processing annotation files...")
            utilities.check_output_dir(self.dir_preprocessed_dataset)
            create_annotation_files.add_sex_and_boneage_to_annotation_file(
                dir_cropping_info,
                os.path.join(self.dir_downloaded_dataset, "boneage-training-dataset.csv"),
                train,
                split,
                self.dir_annotations)
            print("Processing annotation files finished.")

            # cropping images and saving them in "train_images" folder
            print("Cropping images...")
            utilities.check_output_dir(self.dir_cropped_images)
            crop_images.crop_images_according_to_annotations(
                self.dir_annotations,
                os.path.join(self.dir_downloaded_dataset, "boneage-training-dataset"),
                self.dir_cropped_images)
            print("Cropping images finsished.")

        self.file_list = utilities.get_list_of_files(self.dir_cropped_images)
        self.number_of_images = len(self.file_list)

    def __prepare_annotations_working_with_original_dataset(self, root, lr_ann_file_path, train, split, Nmax):
        # create annotation file with Nmax images

        if train:
            print("preparing train annoation file")
        else:
            print("preparing test annotation file")

        # open original train annotation file (csv)
        path_original_ann_file = os.path.join(root, "downloaded_dataset", "boneage-training-dataset.csv")
        with open(path_original_ann_file, 'r') as f_csv_original:
            df_ann_original = pd.read_csv(f_csv_original, sep=",", dtype={'id' : np.int32, 'boneage' : np.int32, 'male' : object})

        # open lr_ann_file
        with open(lr_ann_file_path, 'r') as f_csv_lr:
            df_ann_lr = pd.read_csv(f_csv_lr, sep=",", index_col=0, header=0 )

        if Nmax:
            flag_image_subset = True
        else:
            flag_image_subset = False

        # consider split factor / Train or Test dataset
        if train:
            line_start = 0
            if not flag_image_subset:
                line_stop = math.floor(df_ann_original.shape[0] * split)
            else:
                line_stop = math.floor(Nmax * split)

        else:
            if not flag_image_subset:
                line_start = math.floor(df_ann_original.shape[0] * split) + 1
                line_stop = df_ann_original.shape[0] - 1
            else:
                line_start = math.floor(Nmax * split) + 1
                line_stop = Nmax - 1

        #print(df_ann_lr)

        ann = []
        for i in range(line_start, line_stop+1):

            curr_df = df_ann_original.iloc[[i]]
            curr_id = curr_df['id'].values[0]

            is_male = (curr_df['male'].values[0].lower().find("true") >= 0)

            # for every line, check if an entry exists in lr_ann_file and is left
            try:
                lr_line = df_ann_lr.loc[curr_id].values
                # look for "False" and "false" but transforming to .lower() first
                is_left = (lr_line[0].lower().find("false") == -1)  # string "false" not found ->  is_left = true
                skip = (lr_line[0].lower().find("skip") >= 0)  # string "skip" found -> skip = true

                if skip:
                    continue

            except KeyError as e:  # no entry found with the current key
                is_left = True

            tmp = {
                'id': curr_id,
                'boneage': curr_df['boneage'].values[0],
                'is_male': is_male,
                'is_left': is_left,
            }

            ann.append(tmp)

        self.annotations = ann

        self.dir_images = os.path.join(self.dir_downloaded_dataset, "boneage-training-dataset")

        self.number_of_images = len(ann)


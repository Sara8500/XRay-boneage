import cv2
import os
import torchvision.transforms as transforms


def get_list_of_files(path_dir):

    return sorted([f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f))])


def check_output_dir(path_dir):

    if not os.path.isdir(path_dir):
        os.makedirs(path_dir)


def check_if_directory_exists(dir):
    if not os.path.isdir(dir):
        print('Directory "{}" does not exist.'.format(dir))


def read_anno(dir_annotation, file_annotation):

    file_name = os.path.join(dir_annotation, file_annotation)[:-4] + '.txt'
    with open(file_name) as file:
        return [line.rstrip('\n') for line in file][0].split(',')


def read_image(dir_images, file_image):

    file_name = os.path.join(dir_images, file_image)
    return cv2.imread(file_name)


def save_image(dir_output, file_image, cropped_image):
    file_name = os.path.join(dir_output, file_image)
    cv2.imwrite(file_name, cropped_image)


def show_tensor(t):

    transform = transforms.Compose([
        transforms.ToPILImage()
        ])

    pil_image = transform(t)
    pil_image.show();


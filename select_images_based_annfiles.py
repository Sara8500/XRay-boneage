import os
import utilities
import shutil


def parse_arguments():
    print('\nParse arguments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_crop', type=str, default=None,
                        help='input image directory')
    parser.add_argument('--dir_images', type=str, default=None,
                        help='annotation directory')
    parser.add_argument('--dir_output', type=str, default=None,
                        help='output directory')

    # Parse and assert arguments
    arguments = parser.parse_args()
    assert arguments.dir_crop is not None
    assert arguments.dir_images is not None

    utilities.check_output_dir(arguments.dir_output)

    # Print arguments
    for keys, values in arguments.__dict__.items():
        print('\t{}: {}'.format(keys, values))

    return argument

def select_images_for_cropping(dir_crop, dir_images, dir_output):
    '''
    copying all .png images that have a corresponding annotation file in dir_crop from dir_images to dir_output
    '''

    list_crop_ann = utilities.get_list_of_files(dir_crop)

    for filename in list_crop_ann:

        filename = filename.split('.')[0] + '.png'

        image_src_path = os.path.join(dir_images, filename)
        image_dst_path = os.path.join(dir_output, "train", filename)

        shutil.copyfile(image_src_path, image_dst_path)

        print(f'copied {image_src_path} to {image_dst_path}')

    print('sucessfully extracted all annotated image data')


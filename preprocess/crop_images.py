import utilities
import argparse
import os

def parse_arguments():
    print('\nParse arguments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_cropping_annotations', type=str, default=None,
                        help='directory with the txt files that contain cropping area of each image')
    parser.add_argument('--dir_images', type=str, default=None,
                        help='path to the original uncropped images')
    parser.add_argument('--dir_out', type=str, default=None,
                        help='path where to save the cropped images')

    # Parse and assert arguments
    arguments = parser.parse_args()
    assert arguments.dir_cropping_annotations is not None
    assert arguments.dir_images is not None
    assert arguments.dir_out is not None

    # Print arguments
    for keys, values in arguments.__dict__.items():
        print('\t{}: {}'.format(keys, values))

    utilities.check_if_directory_exists(arguments.dir_cropping_annotations)
    utilities.check_if_directory_exists(arguments.dir_images)

    return arguments


def crop_images_according_to_annotations(dir_annotations, dir_images, dir_out):

    list_files = utilities.get_list_of_files(dir_annotations)
    utilities.check_output_dir(dir_out)
    print("Cropping and saving {} images.".format(len(list_files)))

    for i in range(0, len(list_files)):
        imagename = list_files[i].replace(".txt", ".png")
        image = utilities.read_image(dir_images, imagename)
        annotation = utilities.read_anno(dir_annotations, list_files[i])

        cropped_image = image[int(annotation[1]):int(annotation[3]), int(annotation[0]):int(annotation[2])]

        utilities.save_image(dir_out, imagename, cropped_image)


def main():
    args = parse_arguments()
    crop_images_according_to_annotations( args.dir_cropping_annotations, args.dir_images, args.dir_out)


if __name__ == '__main__':
    main()


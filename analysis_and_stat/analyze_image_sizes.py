import utilities
import os
from PIL import Image
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

dimension_list = []

for f in all_images:
    tmp_path = os.path.join(args.training_dataset, f)
    img = Image.open(tmp_path)  # img is PIL image
    tmp_dim = (img.height, img.width) #tuple
    dimension_list.append(tmp_dim)

filename = "all_images_dimensions.csv"
i = 0
with open(os.path.join(args.out_dir,filename), 'w') as f_out:
    f_out.write("ImageID ; height ; width \n")
    for l in dimension_list:
        f_out.write("{} ; {} ; {}\n".format(all_images[i].replace(".png", ""), l[0], l[1]))
        i += 1

dimension_list = sorted(dimension_list)

my_stat = []
current = None
counter = 0

for d in dimension_list:
    if d != current:
        # save
        my_stat.append([current, counter])

        # continue with next tuple
        current = d
        counter = 1
    else:
        counter += 1

# save last entry
my_stat.append([current, counter])


print("Total number of pictures: ", len(all_images))
print("Most frequent image dimension [(height,width), number_occurences]   ",  max(my_stat, key=lambda x: x[1]))

#save to file

filename = "statistic_image_dimensions.csv"

print("saving result to file: {}".format(os.path.join(args.out_dir,filename)))
with open(os.path.join(args.out_dir,filename), 'w') as f_out:
    for l in my_stat:
        f_out.write("{} ; {}\n".format(l[0], l[1]))

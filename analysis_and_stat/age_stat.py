import pandas as pd
import matplotlib.pyplot as plt
import argparse


def parse_arguments():
    print('\nParse arguments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv_path', type=str, default=None,
                        help='path to the csv directory')
    arguments = parser.parse_args()
    return arguments


args = parse_arguments()

df_ann = pd.read_csv(args.dataset_csv_path )
age = df_ann.boneage
boneage_list = age.values
print(boneage_list)
p = df_ann['boneage'].max()
q = df_ann['boneage'].min()


print(p)
print(q)

plt.figure()
plt.hist(df_ann['boneage'])
plt.show()

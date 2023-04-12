import random
import csv
from tqdm import tqdm
import json

 
def create_patches(files, patches_per_slide):
    patches = []
    for i in files:
        patches += patches_per_slide * [i]
    random.shuffle(patches)
    return patches

def load_slides(set, equal_sampling = True, target_folder=None, csv_path=None, annotations_file=None, label_dict = None, dataset_type = None):
    train_files = []
    valid_files = []
    test_files = []

    with open(csv_path, newline='') as f, open(annotations_file) as d:
        reader = csv.DictReader(f, delimiter=';')
        data = json.load(d)
        for row in tqdm(list(reader)):
            if row["Dataset"] == "train" and set.__contains__("train"):
                train_files.append(target_folder+"/"+row["Slide"])

            elif row["Dataset"] == "val" and set.__contains__("valid"):
                valid_files.append(target_folder+"/"+row["Slide"])

            elif row["Dataset"] == "test" and set.__contains__("test"):
                test_files.append(target_folder+"/"+row["Slide"])
            else:
                pass

    return train_files, valid_files, test_files


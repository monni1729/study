import json
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)

    args = parser.parse_args()

    return args

def data_parsing(data_root):
    folder_list = list(os.listdir(data_root))
    annotation = []
    label_mapping = {}
    label_index = 0
    for category in folder_list:
        if not os.path.isdir(os.path.join(data_root, category)):
            continue
        label_mapping[category] = label_index
        label_index += 1

        img_folder = os.path.join(data_root, category)
        img_folder_list = os.listdir(img_folder)
        for img_filename in img_folder_list:
            temp_dict = {}
            temp_dict['image'] = os.path.join(category, img_filename)
            temp_dict['category'] = category
            annotation.append(temp_dict)

    ## save annotation
    with open(os.path.join(data_root, 'train_annotation.json'), 'w')as json_f:
        json.dump(annotation, json_f, ensure_ascii=False, indent='\t')

    ## save label mapping
    with open(os.path.join(data_root, 'label_mapping.json'), 'w')as json_f:
        json.dump(label_mapping, json_f, ensure_ascii=False, indent='\t')


if __name__ == "__main__":
    args = get_args()
    data_parsing(data_root=args.data_root)

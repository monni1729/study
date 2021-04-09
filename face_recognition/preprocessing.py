import os
import csv
import sys
import json
import random
import skimage.io as io
import albumentations as al


def crop_manual(rd, target_size):
    
    path = "./cropped_images/manual"        
    if not os.path.isdir(path):
        os.mkdir(path)

    child_ids = dict()
    new_annotations = []
    cat_cnt = dict()
    with open('annotations.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for idx, row in enumerate(csv_reader):
            if row[0] not in child_ids:
                child_ids[row[0]] = len(child_ids)
                 
            x1 = int(float(row[6]))
            x2 = int(float(row[7]))
            y1 = int(float(row[8]))
            y2 = int(float(row[9]))
            
            image_dir = os.path.join(rd, row[3])
            img = io.imread(image_dir)
            
            if x2-x1 > y2-y1:
                gap = ((x2-x1) - (y2-y1)) // 2
                y1 = max(y1 - gap, 0)
                y2 = min(y2 + gap , img.shape[0]-1)
            elif x2-x1 < y2-y1:
                gap = ((y2-y1) - (x2-x1)) // 2
                x1 = max(x1 - gap, 0)
                x2 = min(x2 + gap , img.shape[1]-1)
            else:
                pass
            
            img = img[y1:y2, x1:x2, :]
            tf = al.Compose([al.Resize(target_size, target_size)])
            img = tf(image=img)['image']
            fname = os.path.join('/home/taylor/pipeline/cropped_images/manual/', 'c' + str(idx) + '.jpg')
            io.imsave(fname, img)
            new_annotations.append({'image_dir' : fname, 'category': child_ids[row[0]]})
           
           # number of each label 
            if child_ids[row[0]] not in cat_cnt.keys():
                cat_cnt[child_ids[row[0]]] = 0
                
            cat_cnt[child_ids[row[0]]] += 1
    
    train_json = []
    val_json = []
    info_json = {'cat_cnt':cat_cnt, 'children_ids': child_ids}
    
    new_cat_cnt = dict()
    for c in cat_cnt:
        new_cat_cnt[c] = 0
    
    for n in new_annotations:
        temp = n['category']
        if new_cat_cnt[temp] < cat_cnt[temp] * 0.8:
            train_json.append(n)
            new_cat_cnt[temp] += 1
        else:
            val_json.append(n)

            
    with open('cropped_images/manual_train_annotations.json', 'w') as jf:
        json.dump(train_json, jf)
    
    with open('cropped_images/manual_val_annotations.json', 'w') as jf:
        json.dump(val_json, jf)
    
    with open('cropped_images/manual_info.json', 'w') as jf:
        json.dump(info_json, jf)
        
def crop_detector():
    pass
    
# 이미지 크기 바꿀때 --> target_size 변경
if __name__ == '__main__':
    crop_manual(rd='/home/taylor/pipeline/raw_images/1', target_size=112)

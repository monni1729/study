import json

with open('cropped_images/manual_annotations.json', 'r') as jf:
    a = json.load(jf)
    
    cat_cnt = dict()
    for b in a:
        if b['category'] not in cat_cnt:
            cat_cnt[b['category']] = 0
        
        cat_cnt[b['category']] += 1
    
    print(cat_cnt)
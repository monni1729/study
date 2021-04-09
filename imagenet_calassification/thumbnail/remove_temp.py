import os

a='''/home/taylor/spo_classification/thumbnail/datasets/original/n00443231_695.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288384_1941.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00439826_3260.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00434075_5063.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00433661_5733.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00442847_7813.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00433802_11513.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00141669_8063.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00442847_10160.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00005787_9282.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00433802_5046.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00442437_1511.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00439826_4766.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00015388_48417.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288000_6865.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288000_7077.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00433802_14773.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00440747_1135.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00443231_6989.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00440747_2960.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00440509_388.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00444651_5106.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288384_2660.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00006484_6099.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00441824_4756.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00015388_12373.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00005787_2594.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00021939_13896.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00440509_2209.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00441073_11410.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00445055_11498.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00440509_722.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00442847_9072.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00440747_789.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00442115_563.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288384_1318.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00445055_5204.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00442115_608.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288000_587.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00434075_639.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00444340_1664.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00021939_4214.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00444651_9370.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00440509_859.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288000_1916.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00005787_4736.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00440747_192.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288000_9099.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00005787_2618.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00433661_3703.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00141669_2632.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288384_2979.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00015388_19656.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00444846_6747.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00443692_895.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00326094_3521.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00005787_12404.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00441824_5357.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288384_1272.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288384_1183.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00444937_317.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288000_9740.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288384_1098.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00445055_5033.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00444846_6460.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00443692_791.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00443231_3635.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00444340_543.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00444846_265.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00141669_5061.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00005787_5459.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288000_1793.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00442115_7167.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00444651_13873.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00442115_5109.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00440218_2567.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00433661_5643.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288384_2448.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00141669_6873.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00441824_5506.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00434075_3541.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00443692_1125.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00443231_5605.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00439826_4207.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288384_3266.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00006484_13308.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00440382_5358.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00445055_7142.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00434075_5200.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00442115_3923.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00433802_3239.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00439826_702.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00445055_5867.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00441824_4528.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00445055_10528.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288000_3085.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00326094_3571.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288000_8063.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00141669_6338.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00440509_3902.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00021939_18476.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00015388_19550.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00443231_32.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00443692_912.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00015388_4067.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00443231_447.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288384_2790.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00444340_335.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00433661_4152.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00444937_5091.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00441073_6476.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00288384_3149.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00433802_13304.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00442115_3098.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00433661_2754.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00324978_1472.JPEG
/home/taylor/spo_classification/thumbnail/datasets/original/n00443803_2633.JPEG'''

a = a.split('\n')
sub_name = list()
for b in a:
    sub_name.append(b.split('/')[-1])
    
print(sub_name)
print(len(sub_name))

original = '/home/taylor/spo_classification/thumbnail/datasets/original/'
thumb = '/home/taylor/spo_classification/thumbnail/datasets/thumb/'
print(len(os.listdir(original)))
print(len(os.listdir(thumb)))

for sn in sub_name:
    # os.remove(original+sn)   
    os.remove(thumb+sn)

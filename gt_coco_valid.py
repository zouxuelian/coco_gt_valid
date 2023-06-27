import numpy as np
import json
import cv2
import random
from tqdm import tqdm


def show_skeleton(img,kpts,color=(255,128,128),thr=0.5):
    kpts = np.array(kpts).reshape(-1,3)
    skelenton = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
    [9, 7], [7, 5], [5, 6],
    [6, 8], [8, 10],
]
    points_num = [num for num in range(17)]
    for sk in skelenton:

        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1] , 1]))
        if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0 and kpts[sk[0], 2] > thr and kpts[
            sk[1], 2] > thr:
            cv2.line(img, pos1, pos2, color, 2, 8)
    for points in points_num:
        pos = (int(kpts[points,0]),int(kpts[points,1]))
        if pos[0] > 0 and pos[1] > 0 and kpts[points,2] > thr:
            cv2.circle(img, pos,2,(0,0,255),-1)  # 为肢体点画红色实心圆
    return img


with open("/media/zou/D/dataset/coco/annotations/person_keypoints_val2017.json","r") as load_f:
    load_dict = json.load(load_f)

images=load_dict['images']
annotations=load_dict['annotations']

id=18491
image = cv2.imread('/media/zou/D/dataset/coco/images/val2017/'+str(id).zfill(12)+'.jpg')
# print('image',image)
# for img in tqdm(images):
#     if img['file_name'] == '100002.jpg':
#         print(img['id'])

joints=list()
skeleton_color = [(179, 0, 0), (228, 26, 28), (255, 255, 51),
          (49, 163, 84), (0, 109, 45), (255, 255, 51),
          (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127),
          (217, 95, 14), (254, 153, 41), (255, 255, 51),
          (44, 127, 184), (0, 0, 255)]
for person in tqdm(annotations):
    if person['image_id'] == id:
        print('here')
            # person_1 = np.array(person_mss[person_num]['keypoints']).reshape(-1, 3)
        color=random.choice(skeleton_color)
        show_skeleton(image, person['keypoints'], color=color)
import matplotlib.pyplot as plt
save_path = '/media/zou/D/DSPNet1/visual/gtmulti-person/'
# plt.savefig(save_path + \
#
#             '_id_' + str(id) + \
#             '_' + img_name + '.png',
#             format='png', dpi=100)
# plt.savefig(save_path + 'id_' + str(id) + '.pdf', format='pdf',
#             dpi=100)
# plt.show()
# plt.close()
cv2.imwrite('/media/zou/D/DSPNet1/visual/gtmulti-person/'+'id_'+str(id)+'.png',image)
cv2.imshow('crow_pose', image)
cv2.waitKey()
# cv2.imwrite('crowd_5.png',image)
# cv2.imwrite('/media/zou/D/DSPNet1/visual/gtmulti-person/'+'id_'+'.png',image)

import cv2
import numpy as np
import os
from manual_segment import App
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimgs



num_of_classes = 356
offset = 200
train_valid_dirs = ['']
output_folder = 'background_removed_auto_and_manual/additional_classes' 
input_data_main_dir = 'background_removed_auto'
orig_dir = '../../data_loader/data/site_period_all'
cnt = 0
for dir_ in train_valid_dirs:
    print(dir_)
    dirName = output_folder + '/' + dir_
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    for k in range (num_of_classes):
        print(k+offset)
        orig_name = orig_dir + '/' + dir_ + '/' + str(k + offset)
        outdirName = output_folder + '/' + dir_ + '/' + str(k+offset)
        if not os.path.exists(outdirName):
            os.mkdir(outdirName)

        cur_input_dir = input_data_main_dir + '/' + dir_ + '/' + str(k+offset)
        for filename in os.listdir(cur_input_dir):
            if filename.endswith(".jpg"):
                print(filename)
                if os.path.isfile(outdirName + '/' + filename):
                    cnt += 1
                    continue
                #print(cnt)
                img_orig = cv2.imread(orig_name + '/' + filename)
                img_auto = cv2.imread(cur_input_dir + '/' + filename)
                combined = np.concatenate((img_orig, img_auto), axis=1)
                cv2.imshow('frame', combined)
                k = cv2.waitKey(0)
                if k == ord(' '):
                    #print('auto is ok') copy auto image to target folder
                    shutil.copy(cur_input_dir + '/' + filename, outdirName + '/' + filename)
                    continue
                elif k == ord('s'):
                    shutil.copy(orig_name + '/' + filename, outdirName + '/' + filename)
                else:
                    # manual segment
                    img_out = App().run(orig_name + '/' + filename)
                    cv2.destroyAllWindows()
                    img_out[np.where((img_out == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
                    cv2.imshow('frame', img_out)
                    k = cv2.waitKey(0)
                    cv2.imwrite(outdirName + '/' + filename, img_out)



        #print(k)

#img = remove_background('51/1889_1.jpg')
#cv2.imwrite('dede.jpg', img)
# cv2.imshow('frame', img)
# cv2.waitKey(0)
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def remove_background( img):
    #== Parameters =======================================================================
    BLUR = 5
    CANNY_THRESH_1 =  10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0,0.0,1.0) # In BGR format


    #== Processing =======================================================================

    #-- Read image -----------------------------------------------------------------------
    #img = cv2.cvtColor(img, cv2.COLOR_)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


    #-- Edge detection -------------------------------------------------------------------
    #blur = cv2.blur(gray,(BLUR,BLUR))
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    #cv2.imshow('frame', edges)
    #cv2.waitKey(0)
    edges = cv2.dilate(edges, None)
    #cv2.imshow('frame', edges)
    #cv2.waitKey(0)
    edges = cv2.erode(edges, None)
    #cv2.imshow('frame', edges)
    #cv2.waitKey(0)


    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Previously, for a previous version of cv2, this line was:
    #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        #print(c.shape)
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape,dtype=np.uint8)
    #cv2.fillConvexPoly(mask, max_contour[0], (255))
    #cv2.convexHull(max_contour[0], max_contour[0]);
    cv2.fillPoly(mask, [max_contour[0]], (255))
    #res = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow('frame', mask)
    #cv2.waitKey(0)


    aa = np.where(mask[..., None] == 255, img, [255, 255, 255])
    aa = aa.astype(np.uint8)

    cv2.drawContours(img, contour_info[0], 0, (0, 255, 0), 3)
    #cv2.imshow('frame', img)
    #cv2.waitKey(0)
    return aa

num_of_classes = 356
offset = 200
train_valid_dirs = ['']
output_folder = 'background_removed_auto'
input_data_main_dir = '../../data_loader/data/site_period_all'

for dir_ in train_valid_dirs:
    print(dir_)
    dirName = output_folder + '/' + dir_
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    for k in range (num_of_classes):
        print(k+offset)

        outdirName = output_folder + '/' + dir_ + '/' + str(k+offset)
        if not os.path.exists(outdirName):
            os.mkdir(outdirName)

        cur_input_dir = input_data_main_dir + '/' + dir_ + '/' + str(k+offset)
        for filename in os.listdir(cur_input_dir):
            if filename.endswith(".jpg"):
                if os.path.isfile(outdirName + '/' + filename):
                    continue
                #filename =  '5547_1.jpg'
                #print(cur_input_dir + '/' + filename)
                img = cv2.imread(cur_input_dir + '/' + filename )
                try:
                    img_out = remove_background(img)
                except:
                    print(filename)
                    raise
                cv2.imwrite(outdirName +'/' + filename,img_out)
                #print(outdirName +'/' + filename)
                #cv2.imwrite(outdirName + '/bgr' +filename , img)
                # f, (ax1, ax2) = plt.subplots(1, 2)
                # ax1.get_yaxis().set_visible(False)
                # ax1.get_xaxis().set_visible(False)
                # ax2.get_xaxis().set_visible(False)
                # ax2.get_yaxis().set_visible(False)
                # ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # ax2.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
                # plt.show()
                # #plt.savefig(outdirName+ '/' + filename + '.png')
                # plt.show()
                # #plt.close()
                # np.testing.assert_equal(0,1)



        #print(k)

#img = remove_background('51/1889_1.jpg')
#cv2.imwrite('dede.jpg', img)
# cv2.imshow('frame', img)
# cv2.waitKey(0)
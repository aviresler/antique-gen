import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

main_dir = '../../data_loader/data/data_16_6/site_period_top_200_bg_removed/'
input_sets = ['train/', 'valid/']
output_sets = ['train_resize/', 'valid_resize/']

for mm, se in enumerate(input_sets):
    for cls in range(200):
        print(cls)
        input_dir = main_dir + se + str(cls)
        for k, filename in enumerate(os.listdir(input_dir)):
            if filename.endswith(".jpg"):
                im = Image.open(os.path.join(input_dir, filename))
                width, height = im.size
                min_size = np.min(im.size)
                max_size = np.max(im.size)
                aspect_ratio = max_size/min_size

                # resize so the smallest dimension will have 448 pixels
                if width >= height:
                    new_width = 448
                    new_height = int(448/aspect_ratio)
                    new_image = im.resize((new_width,new_height))
                else:
                    new_height = 448
                    new_width = int(448/aspect_ratio)
                    new_image = im.resize((new_width,new_height))

                # pad so there would be 1x1 aspect ratio
                max_size = 448
                temp_image = Image.new('RGB', (448, 448), (255, 255, 255))
                x_margin = int((max_size - new_width)*0.5)
                y_margin = int((max_size - new_height)*0.5)
                temp_image.paste(new_image, (x_margin, y_margin))

                # resize to uniform size of 672x672
                #out_image = temp_image.resize((672, 672))
                out_path = main_dir + output_sets[mm] + str(cls) + '/' + filename
                if not os.path.isdir(main_dir + output_sets[mm] + str(cls)):
                    os.mkdir(main_dir + output_sets[mm] + str(cls))
                temp_image.save(out_path, quality=100)
                #np.testing.assert_equal(0,1)




                #print(aspect_ratio)

                #new_image.show()
                #print(new_image.size)

            #np.testing.assert_equal(0,1)



# widths = np.zeros(14445,dtype=np.int)
# heights = np.zeros(14445,dtype=np.int)
# cnt = 0
# for k,filename in enumerate(os.listdir(directory)):
#     if filename.endswith(".jpg"):
#         im = Image.open(os.path.join(directory, filename))
#         width, height = im.size
#         widths[k] = width
#         heights[k] = height
#
#         cnt = cnt + 1
#         #print(os.path.join(directory, filename))
#
# # _ = plt.hist(widths, bins='auto')
# # print(np.min(widths))
# # print(np.max(widths))
# # plt.show()
#
# # creating a object
# image = Image.open('../../data_loader/data/antiques_all_images/102_3.jpg')
# MAX_SIZE = (672, 672)
#
# image.thumbnail(MAX_SIZE)
#
# # creating thumbnail
# image.save('pythonthumb2.jpg')
# print(image.size)
# image.show()

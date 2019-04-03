import wget
import requests
import re
import csv

offset = 5768
num_of_images = 7930-offset+1

#num_of_images = 2000

# res = requests.get('http://www.antiquities.org.il/t/item_en.aspx?CurrentPageKey=2070')
# m = re.search('href="(images[^ ]*jpg)"', res.text)
# print(m.group(1))
# text_file = open("Output11.txt", "w")
# text_file.write(res.text)
# text_file.close()

# res = requests.get('http://www.antiquities.org.il/t/item_en.aspx?CurrentPageKey=15')
# m = re.search('href="(images[^ ]*jpg)"', res.text)
# print(m.group(1))
# text_file = open("Output2.txt", "w")
# text_file.write(res.text)
# text_file.close()

object_name = []
period = []
site = []
material = []
hieght = []
length = []
width = []
index = []
diameter = []

try:
    for k in range(num_of_images):
        res = requests.get('http://www.antiquities.org.il/t/item_en.aspx?CurrentPageKey={}'.format(offset+k))
        # get image link
        m = re.findall('href="(images[^ ]*jpg)"', res.text)
        set_image = set(m)
        print(set_image)
        if m is not None:
            for kk in range(len(set(m))):
                # download image
                wget.download('http://www.antiquities.org.il/t/{}'.format(set_image.pop()),out='{}_{}.jpg'.format((offset+k),kk))
                index.append(offset+k)
        else:
            index.append(-1)

        # object name
        #print(res.text)
        m = re.search('Object\'s Name[ ]?: <\/b>([^<]*)<b', res.text)
        if m is not None:
            str1111 = m.group(1)
            str1111 = str1111.replace('&nbsp', '')
            object_name.append(str1111)
            print(str1111)
        else:
            object_name.append("none")


        # site
        m = re.search('Site[ ]?: <\/b>([^<]*)<b', res.text)
        if m is not None:
            str1111 = m.group(1)
            str1111 = str1111.replace('&nbsp', '')
            site.append(str1111)
            print(str1111)
        else:
            site.append('none')

        # period
        m = re.search('Period[ ]?: <\/b>([^<]*)<b', res.text)
        if m is not None:
            str1111 = m.group(1)
            str1111 = str1111.replace('&nbsp', '')
            period.append(str1111)
            print(str1111)
        else:
            period.append('none')

        # Material
        m = re.search('Material[ ]?: <\/b>([^<]*)<b', res.text)
        if m is not None:
            str1111 = m.group(1)
            str1111 = str1111.replace('&nbsp', '')
            material.append(str1111)
            print(str1111)
        else:
            material.append('none')



        # Height
        m = re.search('Height[ ]?: <\/b>([^<]*) cm<b', res.text)
        if m is not None:
            str1111 = m.group(1)
            str1111 = str1111.replace('&nbsp', '')
            hieght.append(str1111)
            print(str1111)
        else:
            hieght.append('none')


        # Length
        m = re.search('Length[ ]?: <\/b>([^<]*) cm<b', res.text)
        if m is not None:
            str1111 = m.group(1)
            str1111 = str1111.replace('&nbsp', '')
            length.append(str1111)
            print(str1111)
        else:
            length.append('none')


        # Wide
        m = re.search('Wide[ ]?: <\/b>([^<]*) cm<b', res.text)
        if m is not None:
            str1111 = m.group(1)
            str1111 = str1111.replace('&nbsp', '')
            width.append(str1111)
            print(str1111)
        else:
            width.append('none')

        # Diameter
        m = re.search('Diameter[ ]?: <\/b>([^<]*) cm<b', res.text)
        if m is not None:
            str1111 = m.group(1)
            str1111 = str1111.replace('&nbsp', '')
            diameter.append(str1111)
            print(str1111)
        else:
            diameter.append('none')

        print(offset+k)

finally:

    res = [object_name, site,  period, material, hieght, length, width, diameter]

    with open('../info/object_name.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in object_name:
            writer.writerow([val])

    with open('../info/site.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in site:
            writer.writerow([val])

    with open('../info/period.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in period:
            writer.writerow([val])

    with open('../info/material.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in material:
            writer.writerow([val])

    with open('../info/hieght.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in hieght:
            writer.writerow([val])

    with open('../info/length.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in length:
            writer.writerow([val])

    with open('../info/width.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in width:
            writer.writerow([val])

    with open('../info/index.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in index:
            writer.writerow([val])

    with open('../info/diameter.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in diameter:
            writer.writerow([val])

    # #Assuming res is a list of lists
    # with open('listss.csv', "w") as output:
    #     writer = csv.writer(output, lineterminator='\n')
    #     writer.writerows(res)


    #text_file = open("Output.txt", "w")
    #text_file.write(res.text)
    #text_file.close()

    #wget.download('http://www.antiquities.org.il/t/images/600/1297683.jpg')

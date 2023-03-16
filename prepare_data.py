import os 
import numpy as np
import cv2

path_train = './dataset'
output_path = './new_train'

train_dir = os.listdir(path_train)

labels_train = []
cat = 0 # brigde crack
bbox_ = []
_width, _height = 448, 448
for sample in train_dir:

    if sample.endswith('.txt'):

        with open(os.path.join(path_train, sample), 'r') as f:

            img=cv2.imread(os.path.join(path_train, sample.replace('.txt', '.jpg')))


            lines = f.readlines()
            if len(lines) == 0:
                continue
            label = ["" for x in range((len(lines)*5 + 1))]

            for idx, line in enumerate(lines):

                bbox = line.split(' ')[1:] # x1, y1 coordinates of the upper left corner of the bounding box, x2, y2 coordinates of the lower right corner of the bounding box, class number
                bbox[-1] = bbox[-1].replace('\n', '')

                x, y, w, h = 0.0, 0.0, 0.0, 0.0

                x = float(bbox[0]) * _width
                y = float(bbox[1]) * _height
                w = min(float(bbox[2]), 1.0) * _width
                h = min(float(bbox[3]), 1.0) * _height

                x_max = min(x + (w/2), float(_width))
                y_max = min(y + (h/2), float(_height))

                x_min = max(x - (w/2), 0.0)
                y_min = max(y - (h/2), 0.0)

                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

                bbox=[str(x_min), str(y_min), str(x_max), str(y_max)]

                label[idx*5 + 1:idx*5 + 5] = bbox
                label[idx*5 + 5] = '0' #category
        
        label[0] = '{}/{}.jpg'.format(output_path, sample.split('.')[0])
        cv2.imwrite('prueba.jpg', img)
    
        labels_train.append(label)
    
    # else:
    #     img=cv2.imread(os.path.join(path_train, sample))
    #     sample=sample.split('.')[0] + '.jpg'
    #     cv2.imwrite(os.path.join(output_path, sample), img)


with open('train.txt', 'w') as f:
    for line in labels_train:
        line_str=''
        for i, l in enumerate(line):
            if i == 0 or i%5 == 0:
                line_str += l + ' '
            else:
                line_str += l + ','
        f.write(line_str)
        f.write('\n')







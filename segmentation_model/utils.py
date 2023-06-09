
import cv2
import numpy as np

# code from https://github.com/MrGiovanni/UNetPlusPlus
from unet_pp.segmentation_models import Unet, Xnet


def build_model(input_size, backbone='resnet50', num_class=2, model_path=None, use_unet=False):
    class_num, activation_type = (num_class, 'softmax') if num_class > 2 else (1, 'sigmoid')
    model_arg = dict(input_shape=input_size, backbone_name=backbone, encoder_weights='imagenet',
                     decoder_block_type='transpose',
                     classes=class_num,
                     activation=activation_type)

    model = Unet(**model_arg) if use_unet else Xnet(**model_arg)
    if model_path:
        model.load_weights(model_path)

    return model

def cv2FindContours(image):
    if cv2.__version__[0] == '4':
        ctrs, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, ctrs, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return ctrs

def read_img_file(path):
    try:
        with open(path, 'rb') as img_file:
            bytes = img_file.read()
            nparr = np.fromstring(bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
    except:
        return None

def save_img_file(path, img):
    try:
        with open(path, 'wb') as img_file:
            data = cv2.imencode('.jpg', img)[1]
            img_file.write(data)
    except:
        return None

def get_points(x1, y1, x2, y2, n):
    points_list = []
    for i in range(1, n):
        x = x1 + int(i * (x2 - x1) / n)
        y = y1 + int(i * (y2 - y1) / n)
        points_list.append([x, y])

    return points_list

def draw_points_between_2points(img, st_x, st_y, end_x, end_y, point_size, color, gap):
    tmp = (end_x - st_x) ** 2 + (end_y - st_y) ** 2
    tmp = tmp ** 0.5

    count = int(tmp // gap)

    if count < 1:
        return [st_x, st_y]
    elif count == 1:
        points_list = [[end_x, end_y]]
    else:
        points_list = get_points(st_x, st_y, end_x, end_y, count)

    for k in range(0, len(points_list)):
        cv2.circle(img, tuple(points_list[k]), point_size, color, -1)

    return points_list[-1]

def draw_outline_points(img, ctrs, color=(255, 0, 0), point_size=2, gap=7):
    for i in range(len(ctrs)):
        draw_point = ctrs[i][0][0].tolist()

        for j in range(1, len(ctrs[i])):
            x = ctrs[i][j][0][0]
            y = ctrs[i][j][0][1]
            draw_point = draw_points_between_2points(img, draw_point[0], draw_point[1], x, y, point_size, color, gap)

        x = ctrs[i][0][0][0]
        y = ctrs[i][0][0][1]
        draw_points_between_2points(img, draw_point[0], draw_point[1], x, y, point_size, color, gap)

def area_cal(contour):
    area = 0
    for i in range(len(contour)):
        area += cv2.contourArea(contour[i])

    return area

def area_sort_key(elem):
    return area_cal([elem])


def sparse_get_point(st_x, st_y, end_x, end_y, gap=7):
    distance = (end_x - st_x) ** 2 + (end_y - st_y) ** 2
    distance = distance ** 0.5

    count = int(distance // gap)
    count_f = distance / gap

    if count < 1:
        return None, [st_x, st_y], count_f
    elif count == 1:
        ps = [[end_x, end_y]]
    else:
        ps = get_points(st_x, st_y, end_x, end_y, count)

    return ps, ps[-1], count_f

def sparse_ctrs(in_ctrs, gap=7):
    out_ctrs = []

    for ctrs in in_ctrs:
        sub_out_ctrs = []
        # first point
        start_point = ctrs[0][0].tolist()
        sub_out_ctrs.append(start_point)
        for i in range(1, len(ctrs)):
            end_x = ctrs[i][0][0]
            end_y = ctrs[i][0][1]
            points, start_point, count_f = sparse_get_point(start_point[0], start_point[1], end_x, end_y, gap)
            if points:
                sub_out_ctrs.extend(np.array(points))

        x = ctrs[0][0][0]
        y = ctrs[0][0][1]
        points, start_point, count_f = sparse_get_point(start_point[0], start_point[1], x, y, gap)
        if points:
            sub_out_ctrs.extend(np.array(points))

        out_ctrs.append(np.array(sub_out_ctrs))

    return out_ctrs


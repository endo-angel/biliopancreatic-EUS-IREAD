
import os,time,cv2
import numpy as np
import pandas as pd
from PIL import Image
from utils import *
from dbscan import dbscan_handle

def img_white_count(img):
    white = img[:, :, 0] == 255
    white_pixel_count = len(img[white])
    return white_pixel_count

def xnet_predict_img(model, src_img, model_input_img_size, predict_area_num=1, gray_threshold=127, verbose=1, min_area=None):
    src_img_w = src_img.shape[1]
    src_img_h = src_img.shape[0]

    img = src_img
    img = img[:, :, :: -1]
    img = cv2.resize(img, (model_input_img_size[1], model_input_img_size[0]))
    img = img.astype(np.float64)
    img = img / 255.

    results = model.predict(np.array([img]), verbose=verbose)
    gray_img = results[0][:, :, 0]
    bw_heatmap = np.uint8(255 * gray_img)
    bw_heatmap[bw_heatmap <= gray_threshold] = 0
    bw_heatmap[bw_heatmap > gray_threshold] = 255
    ai_ctrs = cv2FindContours(bw_heatmap.copy())
    ai_ctrs.sort(key=area_sort_key, reverse=True)
    if predict_area_num != -1:
        ai_ctrs = ai_ctrs[:predict_area_num]

    to_del = []
    if min_area:
        if min_area >= 1:
            for ii,ctrs in enumerate(ai_ctrs):
                if area_cal([ctrs]) < min_area:
                    to_del.append(ii)
        else:
            for ii,ctrs in enumerate(ai_ctrs):
                ratio = area_cal([ctrs])/src_img_h/src_img_w
                if ratio < min_area:
                    to_del.append(ii)

    if len(to_del):
        to_del.sort(reverse=True)
        for jj in to_del:
            ai_ctrs.pop(jj)

    x_rate = src_img_w / model_input_img_size[1]
    y_rate = src_img_h / model_input_img_size[0]

    for i in range(len(ai_ctrs)):
        ai_ctrs[i][:, :, 0] = ai_ctrs[i][:, :, 0] * x_rate
        ai_ctrs[i][:, :, 1] = ai_ctrs[i][:, :, 1] * y_rate

    return ai_ctrs

def xnet_predict_file(model, file, model_input_img_size, predict_area_num=1, gray_threshold=127, verbose=1, min_area=None):
    src_img = read_img_file(file)
    ai_ctrs = xnet_predict_img(model, src_img, model_input_img_size, predict_area_num, gray_threshold, verbose, min_area)
    return src_img, ai_ctrs


def predict_file(model, model_input_size, file, result_path,
                 predict_area_num=1, min_area=None, points_gap=5, test_label_path=None):
    file_path, file_name = os.path.split(file)
    src_img, ai_ctrs = xnet_predict_file(model, file, model_input_size, predict_area_num, min_area=min_area)
    src_img_h,src_img_w,_ = src_img.shape

    black_mask_image = Image.new('RGB', (src_img_w, src_img_h), (0, 0, 0))
    black_mask_image = np.array(black_mask_image)

    all_ai_mask_image = black_mask_image.copy()
    all_ai_area = 0

    ai_line_color = (255, 0, 0)
    gt_line_color = (255, 255, 0)
    white_color = (255, 255, 255)

    for item in ai_ctrs:
        sparse_ai_ctrs = sparse_ctrs([item], gap=points_gap)
        sparse_ai_ctrs = np.expand_dims(sparse_ai_ctrs, axis=2)
        ai_mask_image = black_mask_image.copy()
        cv2.fillPoly(ai_mask_image, sparse_ai_ctrs, white_color)
        ai_area = img_white_count(ai_mask_image)
        all_ai_area += ai_area
        all_ai_mask_image += ai_mask_image

    draw_outline_points(src_img, ai_ctrs, color=ai_line_color, gap=points_gap, point_size=2)

    if test_label_path:
        label_img = read_img_file(os.path.join(test_label_path, file_name))
        label_img[label_img >= 127] = 255
        label_img[label_img < 127] = 0
        label_area = img_white_count(label_img)

        label_img_copy = cv2.cvtColor(label_img.copy(), cv2.COLOR_BGR2GRAY)
        label_ctrs = cv2FindContours(label_img_copy)
        draw_outline_points(src_img, label_ctrs, color=gt_line_color, gap=points_gap, point_size=2)
        overlap_img = cv2.bitwise_and(all_ai_mask_image, label_img)
        overlap = img_white_count(overlap_img)
        union_img = cv2.bitwise_or(all_ai_mask_image, label_img)
        union = img_white_count(union_img)

        iou = round(overlap / union, 4)
        dice = round(2 * overlap / (all_ai_area + label_area), 4)

    save_img_file(os.path.join(result_path, file_name), src_img)

    if test_label_path:
        return src_img_w, src_img_h, label_area, all_ai_area, iou, dice
    else:
        return src_img_w, src_img_h, all_ai_area


def predict_path(model, model_input_size, img_path, result_path, label_path, is_chk_dbscan=False):
    test_files = [os.path.join(img_path, f) for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]

    label_area_list = []
    ai_area_list = []
    img_width_list = []
    img_height_list = []
    ai_area_rate_list = []

    iou_list = []
    dice_list = []

    for file in test_files:
        src_img_w, src_img_h, label_area, ai_area, iou, dice = predict_file(model, model_input_size,
                                                                                        file, result_path,
                                                                                        test_label_path=label_path,
                                                                                        is_chk_dbscan=is_chk_dbscan)

        img_width_list.append(src_img_w)
        img_height_list.append(src_img_h)

        label_area_list.append(label_area)
        ai_area_list.append(ai_area)

        img_area = src_img_w * src_img_h
        ai_area_rate_list.append(round(ai_area / img_area, 4))
        iou_list.append(iou)
        dice_list.append(dice)

    df = pd.DataFrame(
        {'filename': test_files, 'width': img_width_list, 'height': img_height_list, 'Ground Truth': label_area_list,
         'AI': ai_area_list, 'ai aera rate': ai_area_rate_list, 'IoU': iou_list, 'dice': dice_list})

    avg_iou = sum(iou_list) / len(test_files)
    avg_iou = round(avg_iou, 4)
    rst_str = str(avg_iou) + time.strftime("_%y%m%d_%H%M%S", time.localtime(time.time()))
    df.to_csv(os.path.join(result_path, 'result_%s.csv' % (rst_str)), encoding='utf-8', index=False, sep=',')


if __name__ == '__main__':
    test_path = r'F:\Biliopancreatic-EUS-IREAD\anatomical_structures_localization\test_data'
    test_img_path = os.path.join(test_path, 'images')
    test_label_path = os.path.join(test_path, 'labels')

    result_path = test_path + '_result_' + time.strftime("%y%m%d_%H%M%S", time.localtime(time.time()))
    os.makedirs(result_path, exist_ok=True)

    model_input_w = 512
    model_input_h = 512
    model_input_size = (model_input_h, model_input_w, 3)
    model_path = r'F:\Biliopancreatic-EUS-IREAD\anatomical_structures_localization\cache\X_21-03-09_19_00_10-0.996.hdf5'
    model = build_model(model_input_size, model_path=model_path)

    predict_path(model, model_input_size, test_img_path, result_path, test_label_path, is_chk_dbscan=True)

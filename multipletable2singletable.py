import os
import json
from PIL import Image
from turtle import left, right
from typing import List
import cv2
import numpy as np
from tqdm import tqdm

img_suffixes = [".png", ".jpg"]
# 获取当前文件父目录
current_path = os.path.dirname(os.path.abspath(__file__))
# 存储目录
save_dir_name = "single_table"
save_dir_path = os.path.join(os.path.dirname(current_path), save_dir_name)
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

def mask_from_polygon(img_shape, polygon: List) -> np.ndarray:
    """Generate mask from polygon.
    Args:
        polygon (List[List]): sorted [[w1, h1], [w2, h2] ...]
    Returns:
        (np.ndarray of bool): cell region mask on original image
    """
    img = np.zeros(img_shape, np.uint8)
    polygon_ndarray = np.array(polygon, dtype="int")
    mask_ = cv2.fillConvexPoly(img, polygon_ndarray, 255)
    return mask_ == 255

def generate_mask_from_cells_mask(img_shape, cells, flag_fill_gap = True):
    """Combine all cell masks to generate mask of table body.

    Args:
        flag_fill_gap (bool, optional): Whether to use morphology methods to fill
        the gaps between cells. Defaults to False.

    Returns:
        Tuple[0] boolean numpy 2-darray: table body mask.
        Tuple[1] uint8 numpy 2-darray: pixel value is corresponding cell id.
    """
    mask = np.zeros(img_shape[:2], dtype="bool")
    for cell in cells:
        mask = mask | cell
    if flag_fill_gap:
        tmp_mask = mask * 255   # convert boolean matrix to image matrix for image proccessing
        tmp_mask = tmp_mask.astype('uint8')
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        tmp_mask = cv2.morphologyEx(tmp_mask, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
        tmp_mask = cv2.dilate(tmp_mask, kernel)
        mask = tmp_mask == 255  # convert back to boolean matrix
    return mask

def region_sorted_bbox(region_mask):
    """Get the sorted bbox of a region mask.

    Args:
        region_mask (np.ndarray): is a uint8 image where pixels in region have 255 value and 0 for the background.
    
    return:
        List[List[int, int]]: a sorted bbox --> [up_left, up_right, down_right, down_left] 
    """
    contour, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contour) > 1:
        print("Warning! Multiple contours!")
    if len(contour) == 0:
        raise Exception("Error! No contour!")
    contour = contour[0]
    bbox = contour_sorted_bbox(contour)
    bbox[0], bbox[2] = np.floor(bbox[0]), np.ceil(bbox[2])
    bbox[1][1], bbox[3][0] = np.floor(bbox[1][1]), np.floor(bbox[3][0])
    bbox[1][0], bbox[3][1] = np.ceil(bbox[1][0]), np.ceil(bbox[3][1])
    return bbox

def contour_sorted_bbox(contour):
    """Get the bbox of a contour.
    Args:
        contour (List[List[int, int], ...]): 
            [[w1, h1], [w2, h2] ...]
    Returns:
        List[List[int, int]]: a sorted bbox --> [up_left, up_right, down_right, down_left] 
    """
    contour = np.array(contour, dtype="float32")
    box_ = cv2.minAreaRect(contour)
    bbox_ = cv2.boxPoints(box_)
    bbox_ = sort_bbox(bbox_)
    return bbox_

def sort_bbox(points: np.ndarray) -> np.ndarray:
    """An unsorted 4-points list of a bbox: [[1, 1], [0, 1], [1, 0], [0, 0]] (in image axis)
        Then the sorted bbox: [[0, 0], [1, 0], [1, 1], [0, 1]] (in image axis)

    Args:
        points (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    ret = np.array(points).reshape(-1, 2)
    x_add_y = ret[:, 0] + ret[:, 1]
    x_sub_y = ret[:, 0] - ret[:, 1]
    idx0, idx1 = np.argmax(x_sub_y), np.argmin(x_add_y)
    idx2, idx3 = np.argmin(x_sub_y), np.argmax(x_add_y)
    return ret[[idx1, idx0, idx3, idx2]]

def walk(top, img_list = []):
    for path, dir_list, file_list in os.walk(top):
        for file_name in file_list:
            if os.path.splitext(file_name)[-1] in img_suffixes:
                img_list.append(os.path.join(path, file_name))
    return img_list

def init_json(**args):
    return args

def init_dict():
    dict = init_json(version = "4.5.9", flags = {}, lineColor = None, fillColor = None, imageData = None, checked = True)
    return dict

def init_shape():
    dict = init_json(shape_type = "polygon", flags = {})
    return dict

def multipletable2singletable(path, save_path):
    # 遍历图片
    image_list = walk(path)
    
    for image_path in tqdm(image_list):
        json_path = os.path.splitext(image_path)[0] + '.json'
        image_name, image_suffix = os.path.splitext(os.path.basename(image_path))
        split = os.path.basename(os.path.dirname(image_path))
        split_dir_path = os.path.join(save_path, split)
        if not os.path.exists(split_dir_path):
            os.makedirs(split_dir_path)

        ## 读取图片
        img = Image.open(image_path)

        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
        img_shape = [json_data["imageHeight"], json_data["imageWidth"]]
        shapes = json_data["shapes"]
        split_shapes = {}
        for shape in shapes:
            if str(shape["group_id"]) not in split_shapes:
                split_shapes[str(shape["group_id"])] = []
            split_shapes[str(shape["group_id"])].append(shape)

        for group_key in split_shapes:
            # 重命名为 {原图片名}_{group_id}.{后缀}
            new_split_name = image_name + '_' + group_key
            new_split_image_path = os.path.join(split_dir_path, new_split_name  + image_suffix)
            new_split_json_path = os.path.join(split_dir_path, new_split_name  + '.json')

            cells_mask = []
            for split_shape in split_shapes[group_key]:
                points = split_shape["points"]
                a_cell_mask_ = mask_from_polygon(img_shape, polygon=points)
                cells_mask.append(a_cell_mask_)
            
            split_mask = generate_mask_from_cells_mask(img_shape, cells_mask).astype("uint8") * 255
            bbox = region_sorted_bbox(split_mask).astype('uint32').tolist()
            left, top = max(min(bbox[0][0], bbox[3][0]), 0), max(min(bbox[0][1], bbox[1][1]), 0)
            right, down = min(max(bbox[1][0], bbox[2][0]), img_shape[1]), min(max(bbox[2][1], bbox[3][1]), img_shape[0])

            # 截取图片并保存
            crop_box = tuple((left, top, right, down))
            newimg = img.crop(crop_box)
            newimg.save(new_split_image_path)

            # 修正shape角点
            for split_shape in split_shapes[group_key]:
                for i in range(0, 4):
                    x, y = max(split_shape["points"][i][0] - left, 0), max(split_shape["points"][i][1] - top, 0)
                    split_shape["points"][i] = [x, y]

            new_dict = init_dict()
            new_dict["imageHeight"] = img.height
            new_dict["imageWidth"] = img.width
            new_dict["imagePath"] = new_split_name  + image_suffix

            new_dict["shapes"] = split_shapes[group_key]

            with open(new_split_json_path, "w") as f:
                json.dump(new_dict, f, indent=2)

if __name__ == "__main__":
    multipletable2singletable(current_path, save_dir_path)

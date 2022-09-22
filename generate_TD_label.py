"""
      Generate mask label of every table based on refined labelme
      cell segmentation annotations and save to png files.

      should input: raw_imgs_path(location of json files)
                    error_save_path(location of error.txt)
                    output_path(save mask label files in this path)
"""

import os
import traceback
import shutil
import json
from typing import List
import glob, cv2
from tqdm import tqdm
import numpy as np


class Cell():
    """A class of table cell stores all possible raw info. on original image.
    """

    def __init__(self,
                 id,
                 img_shape,
                 img_name,
                 mask: np.ndarray = None,
                 polygon=None, ):
        """
        Args:
            id (Integer): differentiate and identify each cell
            img_shape (Tuple): shape of original image (Height, Width)
            mask (np.ndarray of bool): cell region mask on original image
            polygon (List[List]): sorted [[w1, h1], [w2, h2] ...]
        """
        self.img_name = img_name
        self.id = id
        self.img_shape = img_shape
        if mask is None:
            if polygon is not None:
                self.mask: np.ndarray = self._mask_from_polygon(polygon)
            else:
                raise Exception("No enough info. for constructing Cell object!")
        else:
            self.mask: np.ndarray = mask

        if polygon is None:
            if mask is not None:
                self.polygon = self._polygon_from_mask(mask)
                # TODO can add WarpRectContour member
            else:
                raise Exception("No enough info. for constructing Cell object!")
        else:
            self.polygon = polygon

    def _mask_from_polygon(self, polygon: List) -> np.ndarray:
        """Generate mask from polygon.
        Args:
            polygon (List[List]): sorted [[w1, h1], [w2, h2] ...]
        Returns:
            (np.ndarray of bool): cell region mask on original image
        """
        img = np.zeros(self.img_shape, np.uint8)
        polygon_ndarray = np.array(polygon, dtype="int")
        mask_ = cv2.fillConvexPoly(img, polygon_ndarray, 255)
        return mask_ == 255

    def _polygon_from_mask(self, mask) -> List[List]:
        """Generate polygon from mask.
        Args:
            mask (np.ndarray of bool): cell region mask on original image
        Raises:
            Exception: No contour found then polygon will not be generated.
        Returns:
            List[List]: sorted [[w1, h1], [w2, h2] ...]
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        mask = mask.astype('uint8') * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            print("Warning! Multiple contours!" + self.img_name)
        if len(contours) == 0:
            raise Exception("Error! No contour!")

        approx = cv2.approxPolyDP(contours[0], 2, True)
        polygon = (approx.squeeze()).tolist()
        return polygon


class TableBody():
    """A class of table body stores all possible raw info. on original image.
       This class is only for storing. DO NOT implement any heavy model methods in this class !!
    """

    def __init__(self,
                 cells: List,
                 img_shape,
                 origin_img_filename=None,
                 flag_fill_mask_gap=True,
                 flag_colored_mask=False) -> None:
        """
        Args:
            cells (List): a list of all cells
            flag_analyse (bool, optional): whether to generate structure graph and html string. Defaults to False.
        """
        self.cells: List = cells  # a list of all cells
        self.cell_num = len(cells)  # number of cells
        self.img_shape = img_shape  # original image shape (Height, Width)
        self.origin_img_filename = origin_img_filename
        self.flag_fill_mask_gap = flag_fill_mask_gap
        self.mask = self._generate_mask_from_cells_mask(flag_fill_mask_gap)
        self.mask_contour = None

        self.colored_mask = None  # differentiate mask of cells with pixel value of corresponding cell ID. Isn't filled gaps!
        self.flag_colored_mask = flag_colored_mask  # whether to generate colored mask. This costs time!

    def _generate_mask_from_cells_mask(self, flag_fill_gap):
        """Combine all cell masks to generate mask of table body.

        Args:
            flag_fill_gap (bool, optional): Whether to use morphology methods to fill
            the gaps between cells. Defaults to False.

        Returns:
            Tuple[0] boolean numpy 2-darray: table body mask.
            Tuple[1] uint8 numpy 2-darray: pixel value is corresponding cell id.
        """
        mask = np.zeros(self.img_shape[:2], dtype="bool")
        for cell in self.cells:
            mask = mask | cell.mask
        if flag_fill_gap:
            tmp_mask = mask * 255  # convert boolean matrix to image matrix for image proccessing
            tmp_mask = tmp_mask.astype('uint8')
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            tmp_mask = cv2.morphologyEx(tmp_mask, cv2.MORPH_CLOSE, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            tmp_mask = cv2.dilate(tmp_mask, kernel)
            mask = tmp_mask == 255  # convert back to boolean matrix
        return mask


class Tables():
    def __init__(self, tables: List, flags,group_id):
        """Initialize instance from tableBody instance and group id tuple list

        Args:
            tables (List): tableBody instance and group id tuple list, e.g. [(tb1, 0), (tb2, 1), ...]
        """
        self.tables: List = tables
        self.flags = flags
        self.group_id=group_id
        if self.tables:
            self.origin_img_filename = self.tables[0][0].origin_img_filename
            self.img_shape = self.tables[0][0].img_shape

    def generate_labelme_json(self, save_path: str = None, labelme_checked_flag=True) -> str:
        temp = {
            "version": "4.5.9",
            "flags": self.flags,
            "shapes": [],
            "lineColor": [0, 255, 0, 128],
            "fillColor": [255, 0, 0, 128],
            "imagePath": self.origin_img_filename,
            "imageData": None,
            "imageHeight": self.img_shape[0],
            "imageWidth": self.img_shape[1],
            "checked": labelme_checked_flag
        }
        for tb_tuple in self.tables:
            cells = tb_tuple[0].cells
            i=0
            # generate polygons field  w or w/o structure or content
            for index, cell in enumerate(cells):
                id = self.group_id[i]
                label = "table" + str(id)
                shapes_temp = {
                    "label": label,
                    "line_color": None,
                    "fill_color": [255, 0, 0, 0],
                    "points": [],
                    "shape_type": "polygon",
                    "group_id": id
                }
                shapes_temp["points"] = cell.polygon
                temp["shapes"].append(shapes_temp)
                i+=1
        if save_path is not None:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(temp, f, ensure_ascii=False, indent=4)

        return temp


def main(json_path, output_path):
    # load json
    f_name, ext = os.path.splitext(json_path)
    base_name = os.path.basename(f_name)
    with open(json_path, encoding='utf-8') as file:
        labelme_annos = json.load(file)

    # initialize parameters
    flag = labelme_annos["flags"]
    polygons = labelme_annos["shapes"]
    img_filename = labelme_annos["imagePath"]
    image_shape = labelme_annos["imageHeight"], labelme_annos["imageWidth"]
    tb_gid_tuple_list = []
    polygons_groupBy_groupID = {}
    group_id_list = []
    for poly_ in polygons:
        key_ = poly_["group_id"]
        if key_ not in group_id_list:
            group_id_list.append(key_)
        if polygons_groupBy_groupID.get(key_) is None:
            polygons_groupBy_groupID[key_] = []
        polygons_groupBy_groupID[key_].append(poly_)

    cells2: List[Cell] = []
    for group_id, polygons in polygons_groupBy_groupID.items():
        # load cells and initialize TableBody object
        cells: List[Cell] = []
        for idx_, polygon_ in enumerate(polygons):
            cell_ = Cell(idx_,
                         image_shape,
                         img_name=img_filename,
                         polygon=polygon_["points"])
            cells.append(cell_)
        table_body = TableBody(cells,
                               cells[0].img_shape,
                               origin_img_filename=img_filename)
        cell_2 = Cell(group_id,
                      image_shape,
                      img_name=img_filename,
                      mask=table_body.mask)
        cells2.append(cell_2)
    table_body_2 = TableBody(cells2,
                             cells2[0].img_shape,
                             origin_img_filename=img_filename,
                             flag_colored_mask=True)
    tb_gid_tuple_list.append((table_body_2, 0))

    tables = Tables(tb_gid_tuple_list, flags=flag,group_id=group_id_list)

    # generate json files
    save_path = os.path.join(output_path, os.path.basename(json_path))
    tables.generate_labelme_json(save_path=save_path, labelme_checked_flag=labelme_annos["checked"])
    shutil.copy(f_name + ".jpg", output_path)


if __name__ == "__main__":
    ######################### inputs #######################
    raw_imgs_path = "X:/LAB/dataset/CurveTabSet/test/"
    error_save_path = "X:/LAB/dataset/CurveTabSet/errors.txt"
    ######################### outputs #######################
    output_path = "X:/LAB/dataset/CurveTabSet/test2/"
    # whether the file to be written exists

    if os.path.exists(output_path):
        flag_delete_ = input("\033[1;31m%s Exists!! Delete first? yes|no: \033[0m" %
                             os.path.basename(output_path))
        if flag_delete_ == "yes":
            shutil.rmtree(output_path)
        else:
            raise Exception("%s Exists!!" % os.path.basename(output_path))
    os.makedirs(output_path)

    ######################### do the job #######################
    file_paths = glob.glob(
        os.path.join(raw_imgs_path, "*.json"))
    errors = []
    # main loop
    for path_ in tqdm(file_paths, desc="Generating TD label."):
        try:
            main(path_, output_path)
        except Exception as e:
            trace_back = traceback.format_exc()
            print(trace_back)
            errors.append((os.path.basename(path_), trace_back, e))

    ######################### save errors #######################
    if os.path.exists(error_save_path):
        flag_delete_ = input("\033[1;31m%s Exists!! Delete first? yes|no: \033[0m" %
                             os.path.basename(error_save_path))
        if flag_delete_ == "yes":
            os.remove(error_save_path)
        else:
            raise Exception("%s Exists!!" % os.path.basename(error_save_path))
    with open(error_save_path, 'w') as f:
        for idx_, error in enumerate(errors):
            error_filename = error[0]
            error_trace_back_lines = error[1]
            f.write("NO.{0}, Filename: {1}************************\n".format(
                idx_ + 1, error_filename))
            error_trace_back_lines = error_trace_back_lines.split("\n")
            for trace_back_line_ in error_trace_back_lines:
                f.write(trace_back_line_ + "\n")
            f.write("\n")
    #############################################################

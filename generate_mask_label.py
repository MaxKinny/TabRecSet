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
                 mask: np.ndarray = None,
                 polygon=None, ):
        """
        Args:
            id (Integer): differentiate and identify each cell
            img_shape (Tuple): shape of original image (Height, Width)
            mask (np.ndarray of bool): cell region mask on original image
            polygon (List[List]): sorted [[w1, h1], [w2, h2] ...]
        """
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = mask.astype('uint8') * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            print("Warning! Multiple contours!")
        if len(contours) == 0:
            raise Exception("Error! No contour!")

        approx = cv2.approxPolyDP(contours[0], 3.8, True)
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
        if flag_colored_mask:
            self.colored_mask = self.generate_colored_mask()

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

    def generate_colored_mask(self, algorithm="fast"):
        if self.colored_mask is None:
            if algorithm == "slow":
                colored_mask = np.zeros(self.img_shape[:2], dtype="int")
                # cost time! Because this is a naturally heavy task!
                for cell in self.cells:
                    colored_mask = colored_mask + (cell.mask * cell.id)
                self.flag_colored_mask = True
            elif algorithm == "fast":
                colored_mask = self.cells[0].mask
                # id_img_ = colored_mask * self.cells[0].id.number
                id_img_ = colored_mask * self.cells[0].id
                for id_, cell in enumerate(self.cells):
                    if id_ == 0:
                        continue
                    colored_mask = np.logical_xor(colored_mask, cell.mask)
                    id_img_ += cell.mask * cell.id
                    #id_img_ += cell.mask * cell.id.number
                colored_mask = (colored_mask * id_img_).astype("int")
                return colored_mask
            else:
                print("Please give a correct algorithm name!")
                colored_mask = None
            return colored_mask
        else:
            print("Colored mask already exists!!")
            return self.colored_mask

    def get_colored_mask_img(self, flag_visualizeable):
        color = {'0': [255, 0, 0], '1': [255, 255, 0], '2': [0, 0, 255], '3': [255, 0, 255], '4': [0, 255, 0],
                 '5': [0, 255, 255],
                 '6': [255, 125, 0]}
        if self.colored_mask is None:
            print("Colored mask is not generated yet!")
        else:
            if not flag_visualizeable:
                return (self.colored_mask * 1).astype("uint8")
            else:
                color_mask_img = np.zeros((self.img_shape[0], self.img_shape[1], 3), dtype="uint8")
                for channel in range(3):
                    img_ = np.zeros(self.img_shape[:2], dtype="int")
                    for idx, cell in enumerate(self.cells):
                        img_ += cell.mask * color[str(idx)][2 - channel]
                    color_mask_img[..., channel] = img_.astype("uint8")
                return color_mask_img

    def mask_to_pic(self, save_img_path):
        color_mask_img = self.get_colored_mask_img(True)
        cv2.imwrite(save_img_path, color_mask_img)


def main(json_path, output_path):
    # load json
    f_name, ext = os.path.splitext(json_path)
    base_name = os.path.basename(f_name)
    png_name = base_name + ".png"
    with open(json_path, encoding='utf-8') as file:
        labelme_annos = json.load(file)

    # initialize parameters
    polygons = labelme_annos["shapes"]
    img_filename = labelme_annos["imagePath"]
    image_shape = labelme_annos["imageHeight"], labelme_annos["imageWidth"]

    polygons_groupBy_groupID = {}
    for poly_ in polygons:
        key_ = poly_["group_id"]
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
                         polygon=polygon_["points"])
            cells.append(cell_)
        table_body = TableBody(cells,
                               cells[0].img_shape,
                               origin_img_filename=img_filename)
        cell_2 = Cell(group_id,
                      image_shape,
                      mask=table_body.mask)
        cells2.append(cell_2)
    table_body_2 = TableBody(cells2,
                             cells2[0].img_shape,
                             origin_img_filename=img_filename,
                             flag_colored_mask=True)
    table_body_2.mask_to_pic(output_path + png_name)


if __name__ == "__main__":
    ######################### inputs #######################
    raw_imgs_path = "X:/LAB/CurveTabSet/test/"
    error_save_path = "X:/LAB/CurveTabSet/errors.txt"
    ######################### outputs #######################
    output_path = "X:/LAB/CurveTabSet/test_mask/"
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

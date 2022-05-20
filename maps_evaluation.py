import sys
import os
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/lib')
import numpy as np
import cv2 as cv
from scipy import ndimage

filename = sys.argv[1]
is_save_imgs = False
if len(sys.argv) >= 3 and sys.argv[2] == 'save':
    is_save_imgs = True


# ********************************************************* #
# ********************** CORNERS ************************** #
# ********************************************************* #

def corners(img_corners, is_save_imgs):
    # gray scaling
    gray = cv.cvtColor(img_corners, cv.COLOR_BGR2GRAY)
    # cv.namedWindow("gray scaled (corners)", cv.WINDOW_NORMAL)
    # cv.imshow('gray scaled (corners)', gray)

    # searching for contours with no approximation on preprocessed image
    ret, gray_binary = cv.threshold(gray, 210, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(gray_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # coloring all external long contours' neighbor gray cells into black
    for cnt in contours:
        if len(cnt) > 200:
            for pxl in cnt:
                if pxl[0][1] + 1 < len(gray) and gray[pxl[0][1] + 1][pxl[0][0]] == 205:
                    gray[pxl[0][1] + 1][pxl[0][0]] = 0
                if pxl[0][1] - 1 >= 0 and gray[pxl[0][1] - 1][pxl[0][0]] == 205:
                    gray[pxl[0][1] - 1][pxl[0][0]] = 0
                if pxl[0][0] + 1 < len(gray[0]) and gray[pxl[0][1]][pxl[0][0] + 1] == 205:
                    gray[pxl[0][1]][pxl[0][0] + 1] = 0
                if pxl[0][0] - 1 >= 0 and gray[pxl[0][1]][pxl[0][0] - 1] == 205:
                    gray[pxl[0][1]][pxl[0][0] - 1] = 0
                if pxl[0][1] + 1 < len(gray) and pxl[0][0] + 1 < len(gray[0]) and gray[pxl[0][1] + 1][
                    pxl[0][0] + 1] == 205:
                    gray[pxl[0][1] + 1][pxl[0][0] + 1] = 0
                if pxl[0][1] + 1 < len(gray) and pxl[0][0] - 1 >= 0 and gray[pxl[0][1] + 1][pxl[0][0] - 1] == 205:
                    gray[pxl[0][1] + 1][pxl[0][0] - 1] = 0
                if pxl[0][1] - 1 >= 0 and pxl[0][0] + 1 < len(gray[0]) and gray[pxl[0][1] - 1][pxl[0][0] + 1] == 205:
                    gray[pxl[0][1] - 1][pxl[0][0] + 1] = 0
                if pxl[0][1] - 1 >= 0 and pxl[0][0] - 1 >= 0 and gray[pxl[0][1] - 1][pxl[0][0] - 1] == 205:
                    gray[pxl[0][1] - 1][pxl[0][0] - 1] = 0

    # external contours is already processed, now we can clear all unexplored cells
    gray[gray == 205] = 255

    # transforming to float32 for a better result of gaussian laplace filter
    gray = np.float32(gray)
    gray = ndimage.gaussian_laplace(gray, 1, mode='reflect')
    # cv.namedWindow("gray scaled gaussian laplace (corners)", cv.WINDOW_NORMAL)
    # cv.imshow('gray scaled gaussian laplace (corners)', gray)

    # blur
    blur = cv.medianBlur(gray, 3)
    # cv.namedWindow("blur (corners)", cv.WINDOW_NORMAL)
    # cv.imshow('blur (corners)', blur)

    # Harris slam maps evaluation usage
    dst = cv.cornerHarris(blur, 10, 9, 0.2)

    # result is dilated for marking the corners
    dst = cv.dilate(dst, None)
    ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # There may be a bunch of pixels at a corner, we take their centroid
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    print("Corners found:", centroids.size)

    # Now draw them
    for centroid in centroids:
        img_corners = cv.rectangle(img_corners, (int(centroid[0] - 3), int(centroid[1] - 3)),
                                   (int(centroid[0] + 3), int(centroid[1] + 3)), (0, 255, 0), 1)

    # cv.namedWindow("corners", cv.WINDOW_NORMAL)
    # cv.imshow('corners', img_corners)

    if is_save_imgs:
        cv.imwrite(filename.split('.')[0] + '_corners.png', img_corners)


# ********************************************************* #
# ****************** ENCLOSED AREAS *********************** #
# ********************************************************* #

def areas(img_areas, is_save_imgs):
    # gray scaling
    gray = cv.cvtColor(img_areas, cv.COLOR_BGR2GRAY)

    # transform to a binary map
    ret, gray_binary = cv.threshold(gray, 210, 255, cv.THRESH_BINARY)
    # cv.namedWindow("gray scaled binary (areas)", cv.WINDOW_NORMAL)
    # cv.imshow('gray scaled binary (areas)', gray_binary)
    gray_binary = np.uint8(gray_binary)

    # searching for contours with no approximation on preprocessed image
    contours, hierarchy = cv.findContours(gray_binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # counting of all contours that have free cells inside (searching for enclosed areas)
    enclosed_areas_cnt = 0
    area = 0
    main_area = 0
    for cnt in contours:
        # for each pxl of contour find all its neighbor cells that are placed inside the contour
        for pxl in cnt:
            is_enclosed_area = False
            pxl_in_area = []
            if cv.pointPolygonTest(cnt, (int(pxl[0][0] + 1), int(pxl[0][1])), False) == 1:
                pxl_in_area.append((int(pxl[0][0] + 1), int(pxl[0][1])))
            if cv.pointPolygonTest(cnt, (int(pxl[0][0] - 1), int(pxl[0][1])), False) == 1:
                pxl_in_area.append((int(pxl[0][0] - 1), int(pxl[0][1])))
            if cv.pointPolygonTest(cnt, (int(pxl[0][0]), int(pxl[0][1] + 1)), False) == 1:
                pxl_in_area.append((int(pxl[0][0]), int(pxl[0][1] + 1)))
            if cv.pointPolygonTest(cnt, (int(pxl[0][0]), int(pxl[0][1] - 1)), False) == 1:
                pxl_in_area.append((int(pxl[0][0]), int(pxl[0][1] - 1)))

            if len(pxl_in_area):
                for pt in pxl_in_area:

                    # white pxl in contour area means that it's an enclosed area
                    if gray_binary[pt[1]][pt[0]] == 255:  # white in contour area
                        enclosed_areas_cnt += 1
                        area += cv.contourArea(cnt)
                        if cv.contourArea(cnt) > main_area:
                            main_area = cv.contourArea(cnt)

                        # draw the contour
                        rect = cv.minAreaRect(cnt)
                        box = cv.boxPoints(rect)
                        box = np.int0(box)
                        cv.drawContours(img_areas, [box], 0, (0, 0, 255), 1)

                        is_enclosed_area = True
                        break
                if is_enclosed_area:
                    break

    # subtraction map area
    area -= main_area
    print('Enclosed areas found:', enclosed_areas_cnt, 'Their proportion to the whole map:', area / main_area)

    # cv.namedWindow("enclosed areas", cv.WINDOW_NORMAL)
    # cv.imshow('enclosed areas', img_areas)

    if is_save_imgs:
        cv.imwrite(filename.split('.')[0] + '_enclosed_areas.png', img_areas)


# ********************************************************* #
# ******************** PROPORTION ************************* #
# ********************************************************* #

def proportion(img_proportion):
    # gray scaling
    gray = cv.cvtColor(img_proportion, cv.COLOR_BGR2GRAY)
    # cv.namedWindow("gray scaled (proportion)", cv.WINDOW_NORMAL)
    # cv.imshow('gray scaled (proportion)', gray)

    # count the mean value of map without unexplored cells
    occupied_cells_count = gray.size - np.count_nonzero(gray)
    gray_buf = gray.copy()
    gray_buf[gray == 205] = 0
    unexplored_cells_count = gray.size - np.count_nonzero(gray_buf) - occupied_cells_count
    mean = np.sum(gray_buf) / (gray.size - unexplored_cells_count)

    # transform to a binary map
    ret, gray_binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)

    # mark unexplored cells on the binary map
    gray_binary[gray == 205] = 205
    free_cells = np.count_nonzero(gray_binary == 255)
    occupied_cells = np.count_nonzero(gray_binary == 0)
    proportion = occupied_cells / free_cells
    print("Proportion:", proportion)


img_corners = cv.imread(filename)
img_areas = img_corners.copy()
img_proportion = img_corners.copy()

corners(img_corners, is_save_imgs)
areas(img_areas, is_save_imgs)
proportion(img_proportion)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()


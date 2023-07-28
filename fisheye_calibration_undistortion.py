#!/usr/bin/env python3
"""
Thanks to :
https://gist.github.com/mesutpiskin/0412c44bae399adf1f48007f22bdd22d
"""
import cv2
import boxx
from boxx import *
from boxx import np, imread

import json


class FisheyeUndistortion:
    def __init__(self, imgs=None, checkboard=(5, 7), size_mm=25, balance=0.5):
        self.balance = balance
        self.checkboard_imgs = imgs
        if imgs is None:
            return
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
        )
        objp = np.zeros((1, checkboard[0] * checkboard[1], 3), np.float32)
        objp[0, :, :2] = (
            np.mgrid[0 : checkboard[0], 0 : checkboard[1]].T.reshape(-1, 2)
            * size_mm
            / 1000
        )
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        assert all(
            [img.shape == imgs[0].shape for img in imgs]
        ), "All images must share the same size."
        for img in imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(
                gray,
                checkboard,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK
                + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
                imgpoints.append(corners)
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        # rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        # tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            # rvecs,
            # tvecs,
            flags=calibration_flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        )
        print("Found " + str(N_OK) + " valid images for calibration")
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")
        xy = gray.shape[::-1]

        balance = self.balance
        scaled_K = K  # The values of K is to scale with image xyension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
        # This is how scaled_K, xy2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            scaled_K, D, xy, np.eye(3), balance=balance
        )
        self.data = {
            "xy": xy,
            "K": K,
            "D": D,
            "new_K": new_K,
            "scaled_K": scaled_K,
            "balance": balance,
        }
        self._set_map()
        boxx.g()

    def _set_map(self):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.data["scaled_K"],
            self.data["D"],
            np.eye(3),
            self.data["new_K"],
            tuple(self.data["xy"]),
            cv2.CV_16SC2,
        )

        self.map1 = map1
        self.map2 = map2

    def undistort(self, img):
        xy = img.shape[:2][::-1]  # xy is the xyension of input image to un-distort
        assert (
            xy[0] / xy[1] == self.data["xy"][0] / self.data["xy"][1]
        ), "Image to undistort needs to have same aspect ratio as the ones used in calibration"

        undistorted_img = cv2.remap(
            img,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return undistorted_img

    TO_LIST_KEY = ["K", "D", "new_K", "scaled_K"]

    def save(self, jsp="fisheye_calibration_data.json"):
        data = {
            k: v.tolist() if k in self.TO_LIST_KEY else v for k, v in self.data.items()
        }
        with open(jsp, "w") as f:
            json.dump(data, f)
        return jsp

    @classmethod
    def load(cls, jsp="fisheye_calibration_data.json"):
        self = cls()
        data = boxx.loadjson(jsp)
        self.data = {
            k: np.array(data[k]) if k in self.TO_LIST_KEY else v
            for k, v in data.items()
        }
        self._set_map()
        return self


if __name__ == "__main__":
    calibrate_dir = "checkboard_imgs"
    checkboard_imgps = sorted(boxx.glob(f"{calibrate_dir}/*.jpg"))
    checkboard_imgs = [imread(imgp) for imgp in checkboard_imgps]

    fisheye_undistortion = FisheyeUndistortion(
        checkboard_imgs,
    )

    imgps = boxx.glob("distort/*.jpg")
    # imgps = checkboard_imgps[:2]
    imgs = [imread(imgp) for imgp in imgps]
    for img in imgs:
        res = fisheye_undistortion.undistort(img)
        show - res

    fisheye_undistortion.save()
    fisheye_undistortion2 = FisheyeUndistortion.load()
    show - fisheye_undistortion2.undistort(img)

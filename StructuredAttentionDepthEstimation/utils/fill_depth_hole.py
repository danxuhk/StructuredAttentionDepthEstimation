#!/usr/bin/env python -O
# vim: set fileencoding=utf-8 :
# Quick and dirty Python port of fill_depth_colorization.m
# from http://cs.nyu.edu/~silberman/code/toolbox_nyu_depth_v2.zip
from __future__ import print_function

import numpy as np
from itertools import product
import scipy.stats
from scipy.sparse import coo_matrix, dia_matrix
from scipy.sparse import linalg
from skimage import io
import time
import sys
import cv2

def fill_depth_colorization(imgRgb, imgDepth, alpha=1.0):
    """
    Preprocesses the kinect depth image using a gray scale version of the
    RGB image as a weighting for the smoothing. This code is a slight
    adaptation of Anat Levin's colorization code:
    See: www.cs.huji.ac.il/~yweiss/Colorization/
    Args:
      imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
               be between 0 and 1.
      imgDepth - HxW matrix, the depth image for the current frame in
                 absolute (meters) space.
      alpha - a penalty value between 0 and 1 for the current depth values.
    """

    # size = 250
    # imgRgb = imgRgb[:size, :size]
    # imgDepth = imgDepth[:size, :size]
    oldMin = np.nanmin(imgDepth)
    oldMax = np.nanmax(imgDepth)
    oldMean = np.nanmean(imgDepth.flatten())

    knownValMask = ~np.isnan(imgDepth)

    # normalize depth image to (0, 1]
    imgDepth = imgDepth.copy()

    imgDepth -= oldMin
    normMax = np.nanmax(imgDepth)
    imgDepth /= normMax

    imgDepth[~knownValMask] = 0

    H, W = imgDepth.shape
    numPix = H * W

    indsM = np.arange(numPix).reshape(H, W)

    # convert to gray image
    #grayImg = imgRgb.mean(axis=2)
    grayImg = cv2.cvtColor(imgRgb, cv2.COLOR_BGR2GRAY)
    # alternative: use hue instead of luminance
    # grayImg = rgb_to_hsv(imgRgb)[..., 0]

    assert np.isnan(grayImg).sum() == 0

    winRad = 1

    tlen = 0
    absImgNdx = 0
    winPixel = (2 * winRad + 1) ** 2
    cols = np.zeros(numPix * winPixel)
    rows = np.zeros(numPix * winPixel)
    vals = np.zeros(numPix * winPixel)
    gvals = np.zeros(winPixel)

    for absImgNdx, (i, j) in enumerate(product(range(H), range(W))):

        nWin = 0
        for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
            for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                if ii == i and jj == j:
                    continue

                rows[tlen] = absImgNdx
                cols[tlen] = indsM[ii, jj]
                gvals[nWin] = grayImg[ii, jj]

                tlen += 1
                nWin += 1

        #if j == 0 and i % 10 == 0:
        #    print(i, j)

        curVal = grayImg[i, j]
        gvals[nWin] = curVal

        assert np.sum(gvals[:nWin]) > 0, "gvals: %s" % (repr(gvals[:nWin]))

        c_var = np.var(gvals[:nWin])
        # c_var = np.mean((gvals(1:nWin+1)-mean(gvals(1:nWin+1))).^2)

        csig = c_var
        # csig = c_var * 0.6
        # TODO
        # mgv = min((gvals(1:nWin)-curVal).^2)
        # if csig < (-mgv/log(0.01)):
        #     csig=-mgv/log(0.01)

        csig = max(csig, 0.000002)

        # gvals(1:nWin) = exp(-(gvals(1:nWin)-curVal).^2/csig)
        gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
        # gvals(1:nWin) = gvals(1:nWin) / sum(gvals(1:nWin))
        s = gvals[:nWin].sum()
        if s > 0:
            gvals[:nWin] /= s

            s = np.round(gvals[:nWin].sum(), 6)
            assert s == 1, "expected sum to be 1: %.10f" % s

        # vals(len-nWin+1 : len) = -gvals(1:nWin)
        vals[tlen - nWin:tlen] = -gvals[:nWin]

        # Now the self-reference (along the diagonal).
        rows[tlen] = absImgNdx
        cols[tlen] = absImgNdx
        assert vals[tlen] == 0     # not yet set
        vals[tlen] = 1             # sum(gvals(1:nWin))

        tlen += 1

    assert tlen <= numPix * winPixel, "%d > %d" % (tlen, numPix * winPixel)

    A = coo_matrix((vals, (rows, cols)), shape=(numPix, numPix))

    vals = knownValMask.flatten() * alpha

    G = dia_matrix((vals, 0), shape=(numPix, numPix))

    #print("solving…")
    start = time.time()
    new_vals = linalg.spsolve((A + G), vals * imgDepth.flatten())
    #print("solving took %.4f s" % (time.time() - start))
    new_vals.shape = H, W

    # denormalize
    new_vals *= normMax
    new_vals += oldMin

    #print("old min/max/mean:", oldMin, oldMax, oldMean)
    #print("new min/max/mean:", new_vals.min(), new_vals.max(), new_vals.mean())

    return new_vals

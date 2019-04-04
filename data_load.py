# This classs loads face images dataset from
# Face Detection Data Set and Benchmark
# University of Massachusetts - Amherst

import os
import cv2
import math
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed

def set_seed(n=0):
    set_random_seed(n)
    np.random.seed(n)

class FDDB_dataset():

    def __init__(self):
        self.DATASET_FOLDER = 'data'
        self.ELLIPSE_FOLDER = 'data\FDDB-folds'

    def get_ellipse_detail_files(self):
        included_extensions = ['ellipseList.txt']
        file_names = [fn for fn in os.listdir(self.ELLIPSE_FOLDER)
                      if any(fn.endswith(ext) for ext in included_extensions)]
        return file_names

    def get_image_file_name(self, fname):
        return self.DATASET_FOLDER + '/' + fname + '.jpg'

    def clean_txt(self, line):
        line = line.strip('\n')
        line = line.strip()
        line = line.replace('  ', ' ')
        line = line.replace('   ', ' ')
        line = line.replace('    ', ' ')
        return line

    def parse_ellipse_file_content(self, c, det):
        count = 0
        while count<len(c):
            fn = self.clean_txt(c[count])
            n = int(self.clean_txt(c[count+1]))

            ell = []
            for j in range(n):
                ell_txt = self.clean_txt(c[count+2+j])
                ej = ell_txt.split(' ')
                ej = ej[:-1]
                fej = [float(k) for k in ej]
                ell.append(fej)
            det.append([fn, j, ell])

            count += 2+n

        return

    def load_recs(self):
        fns = self.get_ellipse_detail_files()

        recs = []
        for fn in fns:
            fullName = self.ELLIPSE_FOLDER + '\\' + fn
            with open(fullName) as f:
                content = f.readlines()
                self.parse_ellipse_file_content(content, recs)

        return recs

    def rec_to_processed_image(self, rec):
        fname = rec[0]
        fpath = self.get_image_file_name(fname)
        img = cv2.imread(fpath)
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        CMAX = 450
        rdiff = CMAX - grey.shape[0]
        assert rdiff>=0, 'Failed to pad images'
        if rdiff>0:
            padd = np.zeros([rdiff, grey.shape[1]])
            padd = padd.astype('uint8')
            grey = np.vstack((grey, padd))
        cdiff = CMAX - grey.shape[1]
        if cdiff>0:
            padd = np.zeros([grey.shape[0], cdiff])
            padd = padd.astype('uint8')
            grey = np.hstack((grey, padd))

        grey_proc = grey * 1
        grey_proc_01 = grey_proc * 0

        ell = rec[2]

        for ei in ell:
            center = (int(ei[3]), int(ei[4]))
            axes = (int(ei[0]), int(ei[1]))
            angle = math.degrees(ei[2])
            white = (255, 255, 255)

            cv2.ellipse(grey_proc, center=center, axes=axes, angle=angle, startAngle=0, endAngle=360, color=white, thickness=-1)
            cv2.ellipse(grey_proc_01, center=center, axes=axes, angle=angle, startAngle=0, endAngle=360, color=white,
                        thickness=-1)

        # img_hor = np.hstack((grey, grey_proc, grey_proc_01))
        # cv2.imshow('image', img_hor)
        # cv2.waitKey(0)

        return grey, grey_proc, grey_proc_01

    def generate_training_data(self, test_size= 0.1, verbose=0):
        recs = self.load_recs()

        if verbose>0:
            print('Loaded %d records' % len(recs))

        X = []
        y = []

        for rec in recs:
            grey, grey_proc, grey_proc_01 = self.rec_to_processed_image(rec)
            X.append(grey)
            y.append(grey_proc_01)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        if verbose>0:
            print('Loaded %d training data' % len(X_train))
            print('Loaded %d test data' % len(X_test))

        return X_train, X_test, y_train, y_test

    def display_image(self, img1, img2):
        border = np.ones([img1.shape[0], 5]) * 255
        border = border.astype('uint8')

        img_hor = np.hstack((img1, border, img2))
        cv2.imshow('image', img_hor)
        cv2.waitKey(0)


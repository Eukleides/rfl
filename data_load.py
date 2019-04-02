# This classs loads face images dataset from
# Face Detection Data Set and Benchmark
# University of Massachusetts - Amherst

import os
import cv2
import math

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

    def read_face_db_details(self):
        fns = self.get_ellipse_detail_files()

        det = []
        for fn in fns:
            fullName = self.ELLIPSE_FOLDER + '\\' + fn
            with open(fullName) as f:
                content = f.readlines()
                self.parse_ellipse_file_content(content, det)

        return det

    def display_image(self, fdrec):
        fname = fdrec[0]
        fpath = self.get_image_file_name(fname)
        img = cv2.imread(fpath)

        ell = fdrec[2]

        for ei in ell:
            center = (int(ei[3]), int(ei[4]))
            axes = (int(ei[0]), int(ei[1]))
            angle = math.degrees(ei[2])
            white = (255, 255, 255)

            cv2.ellipse(img, center=center, axes=axes, angle=angle, startAngle=0, endAngle=360, color=white, thickness=1)

        cv2.imshow('image', img)
        cv2.waitKey(0)

fd = FDDB_dataset()
fdb = fd.read_face_db_details()
fd.display_image(fdb[1])
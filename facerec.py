import os
import cv2
import sys
import shutil
import random
import numpy as np

'''
Credit to:
Matlab code comments
https://www.wikiwand.com/en/Eigenface#/CITEREFTurkPentland1991
Python implementation
https://github.com/agyorev/Eigenfaces
'''
class Facerec(object):

    person_count = 40
    data_dir = '.'

    # number of faces used for training
    train_faces_count = 6
    # number of faces used for testing
    test_faces_count = 4

    # [h,w,n] = size(yalefaces);
    # n is total training images count
    n = train_faces_count * person_count
    h = 92
    w = 112
    # d is dimension, d = h*w;
    d = h*w

    """
    Constructor for calculating eigenfaces
    """
    def __init__(self, _data_dir ='.', _totalVariance = 0.95):

        self.data_dir = _data_dir
        self.totalVariance = _totalVariance

        # train image id's for every face
        self.training_ids = []

        # % vectorize images
        # x = reshape(yalefaces,[d n]);
        # x = double(x);
        # each row of L represents one train image
        x = np.empty(shape=(self.d, self.n), dtype='float64')

        # load yalefaces
        # !/usr/bin/env python
        current_img = 0
        for person_id in xrange(1, self.person_count + 1):
            # randomly choose 6 photos from each s directory to be training photos
            training_face_ids = random.sample(range(1, 11),
                                              self.train_faces_count)
            # remembering the training id's to get testing id's later
            self.training_ids.append(training_face_ids)

            for train_face_id in training_face_ids:
                # relative path
                path_to_img = os.path.join(self.data_dir,
                                           's' + str(person_id), str(train_face_id) + '.pgm')
                # print '> reading file: ' + path_to_img

                # read a grayscale image
                img = cv2.imread(path_to_img, 0)
                # flatten the 2d image into 1d
                img_col = np.array(img, dtype='float64').flatten()

                # set the cur_img-th column to the current training image
                x[:, current_img] = img_col[:]
                current_img += 1

        # %subtract mean
        # x = bsxfun(@minus, x, mean(x,2));
        # get the mean of all images / over the rows of L
        self.mean_img_col = np.sum(x, axis=1) / self.n

        # subtract from all training images
        for j in xrange(0, self.n):
            x[:, j] -= self.mean_img_col[:]

        # % calculate covariance
        # s = cov(x');
        # computing the covariance matrix as
        # s = x^T*x/n
        s = np.matrix(x.transpose()) * np.matrix(x)
        s /= self.n

        # % obtain eigenvalue & eigenvector
        # [V,D] = eig(s);
        # eigval = diag(D);
        self.V, self.D = np.linalg.eig(s)

        # % sort eigenvalues in descending order
        # eigval = eigval(end:-1:1);
        # V = fliplr(V);
        sort_indices = self.V.argsort()[::-1]
        self.V = self.V[sort_indices]
        self.D = self.D[sort_indices]

        # % show 0th through 15th principal eigenvectors
        # eig0 = reshape(mean(x,2), [h,w]);
        # figure,subplot(4,4,1)
        # imagesc(eig0)
        # colormap gray
        # for i = 1:15
        #     subplot(4,4,i+1)
        #     imagesc(reshape(V(:,i),h,w))
        # end

        # % evaluate the number of principal components needed to represent 95% Total variance.
        # eigsum = sum(eigval);
        # csum = 0;
        eigsum = sum(self.V[:])
        evalues_count = 0
        _totalVariance = 0.0

        # for i = 1:d
        for evalue in self.V:
            evalues_count += 1
            #     csum = csum + eigval(i);
            #     tv = csum/eigsum;
            #     tv = tv + eigval(i)/eigsum;
            _totalVariance += evalue / eigsum
            #     if tv > 0.95
            #         k95 = i;
            if _totalVariance >= self.totalVariance:
                #         break
                break
            #     end;
        # end;

        self.V = self.V[0:evalues_count]
        self.D = self.D[0:evalues_count]
        # change eigenvectors from rows to columns
        self.D = self.D.transpose()
        # left multiply to get the correct evectors
        self.D = x * self.D
        # find the norm of each eigenvector
        norms = np.linalg.norm(self.D, axis=0)
        # normalize all eigenvectors
        self.D = self.D / norms
        # computing the weights
        self.W = self.D.transpose() * x

    """
    Get the closest face index
    """
    def identify(self, path_to_img):
        img = cv2.imread(path_to_img, 0)                                        # read as a grayscale image
        img_col = np.array(img, dtype='float64').flatten()                      # flatten the image
        img_col -= self.mean_img_col                                            # subract the mean column
        img_col = np.reshape(img_col, (self.d, 1))                              # reshape from row vector to col vector

        S = self.D.transpose() * img_col                                        # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

        diff = self.W - S                                                       # finding the min ||W_j - S||
        norms = np.linalg.norm(diff, axis=0)

        closest_face_id = np.argmin(norms)                                      # the id of the minerror face to the sample [0..240)
        return (closest_face_id / self.train_faces_count) + 1                   # return the faceid (1..40)

    """
    Test face recognition with the test faces
    which are the faces that were not picked as the training faces
    """
    def evaluate(self):
        results_file = os.path.join('results', 'facerec_results.txt')           # filename for writing the evaluating results in
        f = open(results_file, 'w')                                             # the actual file

        test_count = self.test_faces_count * self.person_count                  # number of all test images/faces
        test_correct = 0
        for person_id in xrange(1, self.person_count + 1):
            for test_id in xrange(1, 11):
                if (test_id in self.training_ids[person_id-1]) == False:        # we skip the image if it is part of the training set
                    path_to_img = os.path.join(self.data_dir,
                            's' + str(person_id), str(test_id) + '.pgm')        # relative path

                    result_id = self.identify(path_to_img)
                    result = (result_id == person_id)

                    if result == True:
                        test_correct += 1
                        f.write('image: %s\nresult: correct\n\n' % path_to_img)
                    else:
                        f.write('image: %s\nresult: wrong, got %2d\n\n' %
                                (path_to_img, result_id))

        accuracy = float(100. * test_correct / test_count)
        print 'Correct: ' + str(accuracy) + '%'
        f.write('Correct: %.2f\n' % (accuracy))
        f.close()                                                               # closing the file

if __name__ == "__main__":
    if not len(sys.argv) == 2:
        print 'Usage: python 2.7 facerec.py ' \
            + '<att_faces data dir>'
        sys.exit(1)

    if not os.path.exists('results'):                                           # create a folder where to store the results
        os.makedirs('results')
    else:
        shutil.rmtree('results')                                                # clear everything in the results folder
        os.makedirs('results')

    efaces = Facerec(str(sys.argv[1]))                                          # create the Eigenfaces object with the data dir
    efaces.evaluate()                                                           # evaluate our model
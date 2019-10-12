import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_image', type=str)
args = parser.parse_args()

if args.train_image is None:
    exit(1)

class pipeline:
    def __init__(self, train_image):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kp1 = self.orb.detect(train_image, None)
        self.kp1, self.des1 = self.orb.compute(train_image, self.kp1)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
        self.train_image = train_image


    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp2 = self.orb.detect(img, None)
        kp2, des2 = self.orb.compute(img, kp2)
        # matches = self.bf.match(self.des1, des2)
        if len(self.des1) > 0 and len(des2) > 0:
            matches = self.flann.knnMatch(np.float32(self.des1), np.float32(des2), k=2)

            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            src_pts = np.float32([self.kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            if len(src_pts) > 0 and len(dst_pts) > 0:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None and mask is not None and len(M) > 0 and len(mask) > 0:
                    matchesMask = mask.ravel().tolist()
                    h, w = img.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    img = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        return img


    def visualisation(self, img):
        img = self.preprocess(img)
        return img


# main
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    train_image = cv2.imread(args.train_image)  # trainImage
    train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
    train_image = cv2.resize(train_image, (0, 0), fx=0.3, fy=0.3)  # Scale resizing
    # TODO: Find out why it is segfaulting
    my_pipeline = pipeline(train_image)

    while(True):
        ret, frame = cap.read()
        # initialize
        frame_size = frame.shape
        frame_width = frame_size[1]
        frame_height = frame_size[0]

        frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)  # Scale resizing

        thresholds = {}

        visualisation = my_pipeline.visualisation(frame)

        # numpy_horizontal_concat = np.concatenate((frame, visualisation), 1)

        cv2.imshow('image', visualisation)

        cv2.waitKey(1)
        # exit if the key "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

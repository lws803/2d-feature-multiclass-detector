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
        import pdb; pdb.set_trace()
        self.orb = cv2.ORB()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.kp1, self.des1 = self.orb.detectAndCompute(train_image, None)
        self.train_image = train_image


    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp2, des2 = self.orb.detectAndCompute(img, None)
        matches = self.bf.match(self.des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        return cv2.drawMatches(self.img, self.kp1, self.train_image, kp2, matches[:10], flags=2)


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

        numpy_horizontal_concat = np.concatenate((frame, visualisation), 1)

        cv2.imshow('image', numpy_horizontal_concat)

        cv2.waitKey(1)
        # exit if the key "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

import cv2 as cv
import os


def video2ims(src, train_path="images", test_path="test_images", factor=2):
    # os.mkdir(train_path)
    # os.mkdir(test_path)
    frame = 0
    cap = cv.VideoCapture(src)
    counts = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print("number of frames : %d" % counts)
    while True:
        ret, im = cap.read()
        if ret is True:
            if frame < 3600:
                path = train_path
            else:
                path = test_path
            im = cv.resize(im, (w//factor, h//factor))
            cv.imwrite(os.path.join(path, str(frame)+".jpg"), im)
            frame += 1
        else:
            break
    cap.release()


video2ims("F:\Projects\PycharmProjects\opencvtest\detection\data\TownCentreXVID.avi")

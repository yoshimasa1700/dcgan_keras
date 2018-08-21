import cv2
import glob

while(1):
    files = glob.glob("./generated_images/*")
    files.sort()

    image = cv2.imread(files[-1])

    if image is None:
        continue
    image = cv2.resize(image, None, None, 2.0, 2.0)
    cv2.imshow("test", image)
    cv2.waitKey(1000)

import cv2
import glob

while(1):
    files = glob.glob("./generated_images/*")
    files.sort()

    image = cv2.imread(files[-1])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        continue
    image = cv2.resize(image, None, None, 2.0, 2.0)
    cv2.imshow("test", image)
    k = cv2.waitKey(1000)

    print(k)
    if k == 115:
        cv2.imwrite("test.png", image)

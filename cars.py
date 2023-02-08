import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
img=cv2.imread("Cars.jpg")
bbox, label, conf= cv.detect_common_objects(img)
result= draw_bbox(img, bbox, label, conf)

print(f"Number of Cars: {label.count('car')}")

plt.imshow(result)
plt.show()

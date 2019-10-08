import cv2

input_img = cv2.imread('./support_files/test_image.png')
cv2.imshow('Hello Image', input_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


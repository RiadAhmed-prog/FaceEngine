import cv2


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

#path of image URL
imagePath = "known_faces/Riad/R_B.jpg"

image = cv2.imread(imagePath)
image = cv2.resize(image,(224,224))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

fm = variance_of_laplacian(gray)

# text = "Image is not blurred..!"
#
# if fm < 100:
#     text = "Image is blurred..!"
print(fm)
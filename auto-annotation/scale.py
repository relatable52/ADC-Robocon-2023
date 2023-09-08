import cv2 as cv

def savescale(path, save):
    image = cv.imread(path)
    scale = 640/image.shape[0]
    image = cv.resize(image, (0,0), fx=scale, fy=scale)
    print(path)
    cv.imwrite(save, image)

for i in range(1, 301):
    order = str("{:04d}".format(i))
    path = r'gen_test/robocon_' + order + '.jpg'
    save = r'test/robocon_' + order + '.jpg'
    savescale(path, save)
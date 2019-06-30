import cv2
import matplotlib.pyplot as plt

image_filename = "../../competition-data/training/images/satImage_032.png"

img = cv2.imread(image_filename)
img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
img180 = cv2.rotate(img, cv2.ROTATE_180)
img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

index = 1
plt.figure(figsize=(10, 10))

rot_names = ['original', 'rot90', 'rot180', 'rot270']
flip_names = ['horizontal', 'vertical', 'both flip', 'no flip']
for i, curr_rot in enumerate([img, img90, img180, img270]):
    hor = cv2.flip(curr_rot, 0)
    ver = cv2.flip(curr_rot, 1)
    both = cv2.flip(curr_rot, -1)
    non = curr_rot

    title = rot_names[i] + ' - '
    for j, curr_flip in enumerate([hor, ver, both, non]):
        plt.subplot(4, 4, index, title=title + flip_names[j])
        plt.imshow(curr_flip)
        index += 1

plt.show()
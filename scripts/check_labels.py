import cv2
import os

IMG_DIR = "data/processed/images/train"
LBL_DIR = "data/processed/labels/train"

for img_file in os.listdir(IMG_DIR):
    img_path = os.path.join(IMG_DIR, img_file)
    label_path = os.path.join(LBL_DIR, img_file.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w, _ = img.shape

    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())

            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Check", img)

    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()
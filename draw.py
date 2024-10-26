import numpy as np
from numpy import ndarray
import PIL
from PIL import ImageDraw, ImageFont
from PIL.Image import Image
import matplotlib.pylab as plt

def plot_heatmap(image, heatmap, kps, kps_weights):
    for kp_id in range(len(kps_weights)):
        if kps_weights[kp_id] > 0:
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.plot(*kps[kp_id].tolist(), "ro")
            plt.title("image")
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap[kp_id], cmap=plt.cm.Blues)
            plt.colorbar(ticks=[0, 1])
            plt.title(f"kp_id: {kp_id}")
            plt.show()
    
    plt.close("all")


point_color = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
               (240, 2, 127), (240, 2, 127),
               (255, 255, 51), (255, 255, 51),
               (254, 153, 41), (44, 127, 184),
               (217, 95, 14), (0, 0, 255),
               (255, 255, 51), (255, 255, 51), (228, 26, 28),
               (49, 163, 84), (252, 176, 243), (0, 176, 240),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142)]


# COCO 17 points
point_name = ["nose", "left_eye", "right_eye",
              "left_ear", "right_ear",
              "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow",
              "left_wrist", "right_wrist",
              "left_hip", "right_hip",
              "left_knee", "right_knee",
              "left_ankle", "right_ankle"]
def draw_keypoints(img: Image,
                   keypoints: ndarray,
                   scores: ndarray = None,
                   thresh: float = 0.2,
                   r: int = 2,
                   draw_text: bool = False,
                   font: str = 'arial.ttf',
                   font_size: int = 10):

    if isinstance(img, ndarray):
        img = PIL.Image.fromarray(img)

    if scores is None:
        scores = np.ones(keypoints.shape[0])

    if draw_text:
        try:
            font = ImageFont.truetype(font, font_size)
        except IOError:
            font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)
    for i, (point, score) in enumerate(zip(keypoints, scores)):
        if score > thresh and np.max(point) > 0:
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=point_color[i],
                         outline=(255, 255, 255))
            if draw_text:
                draw.text((point[0] + r, point[1] + r), text=point_name[i], font=font)

    return img
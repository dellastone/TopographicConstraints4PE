import scipy.io
import numpy as np

import scipy.io
import numpy as np
def generate_heatmaps(keypoints, heatmap_size=(64, 64), sigma=2.0, num_keypoints=16):
    H, W = heatmap_size
    heatmaps = np.zeros((num_keypoints, H, W), dtype=np.float32)

    for idx in range(num_keypoints):
        x, y, v = keypoints[idx]
        if v < 1:
            continue  # Skip invisible

        x = int(x * W)
        y = int(y * H)

        if not (0 <= x < W and 0 <= y < H):
            continue

        # Create Gaussian
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))  # (H, W)
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))  # (H, W)

        heatmaps[idx] = gaussian  # Assign correctly
    return torch.tensor(heatmaps, dtype=torch.float32)




def load_mpii_annotations(mat_path):
    mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    annolist = mat['RELEASE'].annolist
    img_train = mat['RELEASE'].img_train
    is_train = img_train == 1

    annotations = []
    for i in range(len(annolist)):
        if not is_train[i]:
            continue

        ann = annolist[i]
        img_name = ann.image.name

        if not hasattr(ann, 'annorect'):
            continue

        annorects = ann.annorect
        if not isinstance(annorects, list):
            annorects = [annorects]

        for rect in annorects:
            if not hasattr(rect, 'annopoints'):
                continue

            annopoints = rect.annopoints

            if not hasattr(annopoints, 'point'):
                continue

            keypoints = np.zeros((16, 3))  # 16 keypoints

            points = annopoints.point
            if isinstance(points, np.ndarray):
                points = points.tolist()
            else:
                points = [points]

            for pt in points:
                idx = int(pt.id) - 1
                x = float(pt.x)
                y = float(pt.y)
                keypoints[idx] = [x, y, 1]  # Visibility = 1

            #count the number of visible keypoints
            visible_count = np.sum(keypoints[:, 2] > 0)
            if visible_count < 12:
                continue
            annotations.append({
                'img_path': img_name,
                'keypoints': keypoints
            })

    return annotations


import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
class MPIIDataset(Dataset):
    def __init__(self, data, img_dir, input_size=(256, 256), num_keypoints=16, heatmap_size=(64, 64)):
        """
        Args:
            data (list): List of dictionaries containing image paths and keypoints.
            img_dir (str): Directory containing the images.
            input_size (tuple): Size to which images will be resized.
            num_keypoints (int): Number of keypoints.
            heatmap_size (tuple): Size of the output heatmaps.
        """
        self.data = data
        self.img_dir = img_dir
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data[idx]
        img_path = os.path.join(self.img_dir, ann['img_path'])
        img = cv2.imread(img_path)
        original_h, original_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)

        # Compute scaling factors
        scale_x = self.input_size[0] / original_w
        scale_y = self.input_size[1] / original_h

        # Adjust keypoints to the resized image coordinates
        keypoints = ann['keypoints'].copy()
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        # Normalize keypoints to [0, 1] for heatmap generation
        normalized_keypoints = keypoints.copy()
        normalized_keypoints[:, 0] /= self.input_size[0]
        normalized_keypoints[:, 1] /= self.input_size[1]

        image_tensor = self.transform(img)
        heatmaps = generate_heatmaps(normalized_keypoints, heatmap_size=self.heatmap_size, num_keypoints=self.num_keypoints)

        return image_tensor, heatmaps


import json
import numpy as np
import os

def iou(box, clusters):
    # box: [w, h], clusters: N x 2
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) == 0 or np.count_nonzero(y == 0) == 0:
        return np.zeros(clusters.shape[0])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_

def kmeans(boxes, k, dist=np.median):
    # boxes: N x 2 [w, h]
    box_number = boxes.shape[0]
    distances = np.empty((box_number, k))
    last_nearest = np.zeros((box_number,))
    np.random.seed(42)
    clusters = boxes[np.random.choice(box_number, k, replace=False)]

    while True:
        for i in range(box_number):
            distances[i] = 1 - iou(boxes[i], clusters)

        nearest = np.argmin(distances, axis=1)

        if (last_nearest == nearest).all():
            break

        for cluster in range(k):
            # Nếu cụm trống, khởi tạo lại
            if len(boxes[nearest == cluster]) == 0:
                clusters[cluster] = boxes[np.random.choice(box_number)]
            else:
                clusters[cluster] = dist(boxes[nearest == cluster], axis=0)

        last_nearest = nearest

    return clusters

def get_anchors(annotation_file, image_size=416):
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # Lấy thông tin kích thước gốc của từng ảnh
    image_dims = {}
    for img in data['images']:
        image_dims[img['id']] = (img['width'], img['height'])
    
    boxes = []
    for ann in data['annotations']:
        img_id = ann['image_id']
        w_orig, h_orig = image_dims[img_id]
        
        bbox = ann['bbox'] # [xmin, ymin, xmax, ymax]
        w_box = bbox[2] - bbox[0]
        h_box = bbox[3] - bbox[1]
        
        # Scaling the box as if the image was resized to `image_size x image_size` with letterbox
        scale = min(image_size / w_orig, image_size / h_orig)
        
        w_new = w_box * scale
        h_new = h_box * scale
        boxes.append([w_new, h_new])
        
    boxes = np.array(boxes)
    anchors = kmeans(boxes, k=9)
    # Sort anchors by area
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    
    return anchors

if __name__ == "__main__":
    train_json = os.path.join(os.path.dirname(__file__), "../public/annotations/train.json")
    if os.path.exists(train_json):
        print(f"Calculating anchors for {train_json} with image size 416...")
        anchors = get_anchors(train_json)
        print("Calculated Anchors (sorted by area):")
        anchors = np.round(anchors).astype(int)
        for i, a in enumerate(anchors):
            print(f"Anchor {i+1}: {a[0]}x{a[1]}")
        
        print("\nProposed ANCHORS variable for config.py:")
        print("ANCHORS = [")
        print(f"    [({anchors[6][0]}, {anchors[6][1]}), ({anchors[7][0]}, {anchors[7][1]}), ({anchors[8][0]}, {anchors[8][1]})], # Scale 13x13 (Ảnh 416, stride 32)")
        print(f"    [({anchors[3][0]}, {anchors[3][1]}), ({anchors[4][0]}, {anchors[4][1]}), ({anchors[5][0]}, {anchors[5][1]})], # Scale 26x26 (Ảnh 416, stride 16)")
        print(f"    [({anchors[0][0]}, {anchors[0][1]}), ({anchors[1][0]}, {anchors[1][1]}), ({anchors[2][0]}, {anchors[2][1]})], # Scale 52x52 (Ảnh 416, stride 8)")
        print("]")
    else:
        print(f"File not found: {train_json}")

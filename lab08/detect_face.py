import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    box_scores = np.asarray(box_scores)
    if box_scores.size == 0 or top_k == 0:
        return box_scores[:0]
    if box_scores.ndim != 2 or box_scores.shape[1] != 5:
        raise ValueError(
            "box_scores must have shape (N, 5): x1, y1, x2, y2, score"
        )

    scores = box_scores[:, 4]
    boxes = box_scores[:, :4]
    candidates = np.argsort(scores)
    if candidate_size == 0:
        return box_scores[:0]
    if candidate_size > 0:
        candidates = candidates[-candidate_size:]

    picked = []
    while candidates.size:
        current = candidates[-1]
        picked.append(current)
        candidates = candidates[:-1]

        if (top_k > 0 and len(picked) >= top_k) or candidates.size == 0:
            break

        active_box = boxes[current]
        remaining_boxes = boxes[candidates]
        xmin = np.maximum(remaining_boxes[:, 0], active_box[0])
        ymin = np.maximum(remaining_boxes[:, 1], active_box[1])
        xmax = np.minimum(remaining_boxes[:, 2], active_box[2])
        ymax = np.minimum(remaining_boxes[:, 3], active_box[3])
        intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        active_area = np.maximum(0, active_box[2] - active_box[0]) * np.maximum(
            0, active_box[3] - active_box[1]
        )
        remaining_areas = np.maximum(0, remaining_boxes[:, 2] - remaining_boxes[:, 0]) * np.maximum(
            0, remaining_boxes[:, 3] - remaining_boxes[:, 1]
        )
        union = active_area + remaining_areas - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
        candidates = candidates[iou <= iou_threshold]

    return box_scores[picked]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    original_width, original_height = image.size
    input_image = image.resize((320, 240))
    input_array = np.asarray(input_image, dtype=np.float32)
    input_array = (input_array - 127.0) / 128.0
    input_tensor = np.transpose(input_array, (2, 0, 1))[np.newaxis, ...]

    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=sess_options,
        providers=available_providers,
    )
    input_name = session.get_inputs()[0].name
    outputs = [np.asarray(output) for output in session.run(None, {input_name: input_tensor})]
    output_shapes = [output.shape for output in outputs]
    box_indices = [i for i, output in enumerate(outputs) if output.ndim >= 2 and output.shape[-1] == 4]
    score_indices = [i for i, output in enumerate(outputs) if output.ndim >= 2 and output.shape[-1] == 2]
    if len(box_indices) != 1 or len(score_indices) != 1:
        raise ValueError(
            "Expected ONNX outputs with 4 box coordinates and 2 class scores; "
            f"received output shapes {output_shapes}"
        )

    boxes = outputs[box_indices[0]][0]
    scores = outputs[score_indices[0]][0]
    if scores.ndim == 2:
        scores = scores[:, 1]
    if boxes.ndim != 2 or boxes.shape[1] != 4 or scores.ndim != 1 or len(scores) != len(boxes):
        raise ValueError(
            "ONNX box and score outputs must contain matching detections; "
            f"received boxes {boxes.shape} and scores {scores.shape}"
        )

    confident = scores >= conf_threshold
    box_scores = np.column_stack((boxes[confident], scores[confident]))
    detections = hard_nms(box_scores, iou_threshold=0.3)

    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, score in detections:
        coordinates = (
            int(x1 * original_width),
            int(y1 * original_height),
            int(x2 * original_width),
            int(y2 * original_height),
        )
        draw.rectangle(coordinates, outline="lime", width=3)
        label = f"Face: {score:.1f}"
        label_y = max(0, coordinates[1] - 12)
        draw.text((coordinates[0], label_y), label, fill="lime")

    image.save(output_path)

# Edge detection
def non_maximum_suppression(magnitude, angle):
    magnitude = np.asarray(magnitude)
    angle_degrees = np.rad2deg(angle) % 180
    thinned = np.zeros(magnitude.shape, dtype=float)
    height, width = magnitude.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            direction = angle_degrees[i, j]
            if direction < 22.5 or direction >= 157.5:
                neighbors = magnitude[i, j - 1], magnitude[i, j + 1]
            elif direction < 67.5:
                neighbors = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif direction < 112.5:
                neighbors = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                neighbors = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if magnitude[i, j] >= neighbors[0] and magnitude[i, j] >= neighbors[1]:
                thinned[i, j] = magnitude[i, j]

    return thinned

def hysteresis_thresholding(img, low_thresh, high_thresh):
    img = np.asarray(img)
    strong = img >= high_thresh
    weak = (img >= low_thresh) & ~strong
    edges = np.zeros(img.shape, dtype=np.uint8)
    edges[strong] = 255

    height, width = img.shape
    for i, j in np.argwhere(weak):
        row_start = max(0, i - 1)
        row_end = min(height, i + 2)
        col_start = max(0, j - 1)
        col_end = min(width, j + 2)
        if np.any(strong[row_start:row_end, col_start:col_end]):
            edges[i, j] = 255

    return edges

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    with Image.open(image_path) as source_image:
        grayscale = source_image.convert("L")

    image_array = np.asarray(grayscale, dtype=np.float32)
    smoothed = gaussian_filter(image_array, sigma=1.4)

    sobel_x = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=np.float32,
    )
    sobel_y = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=np.float32,
    )
    gradient_x = convolve(smoothed, sobel_x)
    gradient_y = convolve(smoothed, sobel_y)
    magnitude = np.hypot(gradient_x, gradient_y)
    max_magnitude = magnitude.max()
    if max_magnitude > 0:
        magnitude = magnitude * (255.0 / max_magnitude)

    angle = np.arctan2(gradient_y, gradient_x)
    thinned = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thinned, low, high)
    Image.fromarray(edges, mode="L").save(output_path)

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")
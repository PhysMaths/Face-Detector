# Face Detector — TensorFlow/Keras

A simple end‑to‑end face detector built in a Jupyter notebook. It uses a **VGG16** backbone with two heads: 
a binary classifier (face/no face) and a bounding‑box regressor trained with a custom localization loss. 
Data are collected from a webcam, annotated with **Labelme**, optionally **augmented with Albumentations**, 
then fed into a `tf.data` pipeline.

---

## Features
- 📷 Data collection from a webcam via OpenCV
- 🏷️ JSON annotations created with **Labelme** (top‑left & bottom‑right points)
- 🔁 Train/val/test split with **Albumentations** for heavy augmentation
- 🧠 **VGG16** backbone; dual‑head model (classification + box regression)
- ⚙️ Custom localization loss (L2 on center coords and width/height)
- 🚀 TensorFlow subclassed `Model` (`FaceTracker`) with custom `train_step` / `test_step`
- 💾 Saves trained model to `facetracker.keras`
- 🖼️ Live inference loop draws predictions on frames

---

## Project Structure (expected)
```
FaceDetection.ipynb
data/
  images/                      # raw captures (used originally: /mnt/c/data/images/*.jpg)
  train/
    images/                    # original images
    labels/                    # labelme JSON, 1 per image
  val/
    images/
    labels/
  test/
    images/
    labels/
aug_data/
  train/
    images/                    # augmented images
    labels/                    # JSON with normalized bbox + class
  val/
    images/
    labels/
  test/
    images/
    labels/
```

> In the notebook the raw/aug roots were `/mnt/c/data` and `/mnt/c/aug_data`. 
  Adjust paths as needed if you’re not on Windows WSL (`/mnt/c/...`).

---

## Environment & Requirements

**Python**: 3.9+ recommended

Install with pip:
```bash
pip install tensorflow opencv-python matplotlib albumentations labelme
```

Optional (for GPU):
```bash
# On systems with NVIDIA GPUs, install the appropriate tensorflow version for your CUDA stack
# pip install tensorflow[and-cuda]  # refer to TF docs for exact versions
```

---

## Data Workflow

1. **Capture images** (OpenCV):
   - The notebook opens your webcam and dumps frames into `data/images/` (in-notebook variable: `IMAGES_PATH`).

2. **Annotate with Labelme**:
   - Draw a single bounding box per face (two points: top‑left and bottom‑right).
   - Save as JSON beside each image in the `labels/` folder (the notebook expects one JSON per image).

3. **Split** your dataset into `train/`, `val/`, and `test/` subfolders (each with `images/` and `labels/`).

4. **Augment** (Albumentations):
   - Random crop, flips, RGB shift, gamma, and brightness/contrast.
   - Augmented images & normalized bboxes are written under `/mnt/c/aug_data` mirroring the split.

---

## Model

- **Input size**: 120×120×3 (images are resized and scaled to `[0,1]`).
- **Backbone**: `VGG16` with `include_top=False` and a **GlobalMaxPooling2D**.
- **Heads**:
  - **Classification**: Dense → Dense(1, sigmoid) → predicts face presence.
  - **Regression**: Dense → Dense(4, sigmoid) → predicts normalized `[x_min, y_min, x_max, y_max]`.
- **Losses**:
  - Classification: `BinaryCrossentropy`.
  - Localization: custom L2 on coordinates and on `(w, h)` derived from the predicted box.
- **Training wrapper**: `FaceTracker(Model)` overriding `compile`, `train_step`, `test_step`, `call`.

---

## Training

The notebook builds `tf.data` pipelines from `/mnt/c/aug_data`:
- Resizes to 120×120
- Scales pixels to `[0, 1]`
- Zips images with labels `[class, bbox]`
- Batches of **8** and prefetching

Typical compile and fit (as in the notebook):
```python
opt = tf.keras.optimizers.Adam(learning_rate=0.0001 (Adam))
classloss = tf.keras.losses.BinaryCrossentropy()
# localization_loss is defined in the notebook
model = FaceTracker(build_model())
model.compile(opt=opt, classloss=classloss, localizationloss=localization_loss)
hist = model.fit(train, epochs=20, validation_data=val)
model.save("facetracker.keras")
```

> Adjust `epochs`, learning rate, and batch size based on your hardware and dataset size.

---

## Inference (Webcam)

The notebook includes a live loop that:
- Captures frames from a webcam (`cv2.VideoCapture`)
- Resizes & normalizes a copy for the model
- Runs `model.predict` (or `__call__`) to get `(class_prob, bbox)`
- Scales normalized bbox back to frame size (e.g., 450×450 in the demo)
- Draws the box + label with OpenCV

Minimal example:
```python
import cv2, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("facetracker.keras", compile=False)  # if custom objects needed, pass them

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (120, 120))
    x = img.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    pred_class, pred_bbox = model(x, training=False)
    if float(pred_class[0,0]) > 0.5:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = pred_bbox[0].numpy()
        p1 = (int(x1*w), int(y1*h))
        p2 = (int(x2*w), int(y2*h))
        cv2.rectangle(frame, p1, p2, (0,255,0), 2)
        cv2.putText(frame, f"face: {float(pred_class[0,0]):.2f}", (p1[0], max(0, p1[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Face Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

> If OpenCV fails to open the camera on Linux/WSL, try specifying a backend: `cv2.VideoCapture(0, cv2.CAP_V4L2)` and set FOURCC to MJPG.

---

## Label Format

Each JSON (per image) is expected to provide:
- `class`: `0` or `1` (no face / face)
- `bbox`: normalized `[x_min, y_min, x_max, y_max]` in the augmented set

For raw Labelme JSON, the notebook converts the first shape with two points to normalized coords with respect to image width/height (640×480 in the example).

---

## Tips & Gotchas

- Ensure consistent resolution *during normalization* (the example uses 640×480 for converting to `[0,1]`).
- Only one face per image is assumed in the provided pipeline. Extend your JSON schema & training loop for multi‑face support.
- When loading a saved model with custom losses, pass `custom_objects={'localization_loss': localization_loss}`.
- If you see shape errors, enforce shapes before loss computation (the notebook uses `tf.ensure_shape` for `[None,1]` and `[None,4]`).

---

## Reproducing the Notebook

1. Install dependencies.
2. Open `FaceDetection.ipynb` in Jupyter/VS Code.
3. Follow the cells in order:
   - Capture images → Annotate → Augment → Build datasets → Define model/loss → Train → Save → Inference.
---

## Acknowledgements

- Keras Applications VGG16
- Labelme for lightweight image annotation
- Albumentations for robust image augmentations

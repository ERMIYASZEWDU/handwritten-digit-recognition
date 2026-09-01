import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source})

md(["# Handwritten Digit Recognition"])
md(["## PHASE 1 — Environment Setup"])

code(["!pip install numpy pandas matplotlib pillow scikit-learn tensorflow seaborn streamlit"])

code([
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from PIL import Image, ImageOps, ImageFilter\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "import seaborn as sns\n",
    "\n",
    "np.random.seed(42)\n",
    "tf.random.set_seed(42)\n",
    "\n",
    "IMG_SIZE = 32\n",
    "NUM_CLASSES = 10\n",
    "DATASET_DIR = \"DA\"\n",
    "BATCH_SIZE = 32\n",
    "\n",
    "print(f\"TensorFlow: {tf.__version__}\")",
])

md(["## PHASE 2 — Dataset Preparation"])

code([
    "class_counts = {}\n",
    "for digit in sorted(os.listdir(DATASET_DIR)):\n",
    "    folder = os.path.join(DATASET_DIR, digit)\n",
    "    if not os.path.isdir(folder): continue\n",
    "    image_files = [f for f in os.listdir(folder) if f.lower().endswith((\".jpg\", \".jpeg\", \".png\", \".bmp\"))]\n",
    "    class_counts[digit] = len(image_files)\n",
    "    print(f\"Digit {digit}: {len(image_files)} images\")\n",
    "print(f\"Total: {sum(class_counts.values())} images\")",
])

code([
    "plt.figure(figsize=(10, 5))\n",
    "plt.bar(class_counts.keys(), class_counts.values(), color=\"steelblue\")\n",
    "plt.title(\"Class Distribution\")\n",
    "plt.show()",
])

md(["## PHASE 3 — Image Preprocessing"])

code([
    "def otsu_threshold(img_array):\n",
    "    histogram = np.bincount(img_array.astype(np.uint8).ravel(), minlength=256)\n",
    "    total = img_array.size\n",
    "    sum_total = np.sum(np.arange(256) * histogram)\n",
    "    sum_bg = 0.0\n",
    "    weight_bg = 0\n",
    "    max_variance = 0\n",
    "    threshold = 0\n",
    "    for t in range(256):\n",
    "        weight_bg += histogram[t]\n",
    "        if weight_bg == 0: continue\n",
    "        weight_fg = total - weight_bg\n",
    "        if weight_fg == 0: break\n",
    "        sum_bg += t * histogram[t]\n",
    "        mean_bg = sum_bg / weight_bg\n",
    "        mean_fg = (sum_total - sum_bg) / weight_fg\n",
    "        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2\n",
    "        if variance > max_variance:\n",
    "            max_variance = variance\n",
    "            threshold = t\n",
    "    return threshold\n",
    "\n",
    "def preprocess_image(img_path):\n",
    "    img = Image.open(img_path).convert(\"L\")\n",
    "    img = img.filter(ImageFilter.GaussianBlur(radius=1))\n",
    "    img_array = np.array(img)\n",
    "    thresh = otsu_threshold(img_array)\n",
    "    img_array = np.where(img_array > thresh, 255, 0).astype(np.uint8)\n",
    "    if np.mean(img_array) < 128:\n",
    "        img_array = 255 - img_array\n",
    "    img = Image.fromarray(img_array)\n",
    "    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)\n",
    "    return np.array(img, dtype=np.float32) / 255.0\n",
    "\n",
    "test_path = os.path.join(DATASET_DIR, \"0\", os.listdir(os.path.join(DATASET_DIR, \"0\"))[0])\n",
    "print(f\"Shape: {preprocess_image(test_path).shape}\")",
])

code([
    "images = []\n",
    "labels = []\n",
    "for digit in sorted(os.listdir(DATASET_DIR)):\n",
    "    folder = os.path.join(DATASET_DIR, digit)\n",
    "    if not os.path.isdir(folder): continue\n",
    "    for f in os.listdir(folder):\n",
    "        if not f.lower().endswith((\".jpg\", \".jpeg\", \".png\", \".bmp\")): continue\n",
    "        try:\n",
    "            images.append(preprocess_image(os.path.join(folder, f)))\n",
    "            labels.append(int(digit))\n",
    "        except: pass\n",
    "images = np.expand_dims(np.array(images, dtype=np.float32), axis=-1)\n",
    "labels = np.array(labels, dtype=np.int32)\n",
    "print(f\"Images: {images.shape}, Labels: {labels.shape}\")",
])

code([
    "fig, axes = plt.subplots(2, 5, figsize=(12, 5))\n",
    "for i in range(10):\n",
    "    idx = np.where(labels == i)[0][0]\n",
    "    r, c = divmod(i, 5)\n",
    "    axes[r, c].imshow(images[idx].squeeze(), cmap=\"gray\")\n",
    "    axes[r, c].set_title(f\"Digit {i}\")\n",
    "    axes[r, c].axis(\"off\")\n",
    "plt.suptitle(\"Preprocessed Images\")\n",
    "plt.tight_layout()\n",
    "plt.show()",
])

md(["## PHASE 4 — Dataset Splitting"])

code([
    "X_train_val, X_test, y_train_val, y_test = train_test_split(\n",
    "    images, labels, test_size=0.2, random_state=42, stratify=labels)\n",
    "X_train, X_val, y_train, y_val = train_test_split(\n",
    "    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)\n",
    "print(f\"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}\")",
])

md(["## PHASE 5 — Data Augmentation"])

code([
    "data_augmentation = keras.Sequential([\n",
    "    layers.RandomRotation(0.08),\n",
    "    layers.RandomTranslation(0.08, 0.08),\n",
    "    layers.RandomZoom(0.08),\n",
    "], name=\"data_augmentation\")\n",
    "\n",
    "def prepare_dataset(X, y, augment=False):\n",
    "    ds = tf.data.Dataset.from_tensor_slices((X, y))\n",
    "    if augment:\n",
    "        ds = ds.shuffle(buffer_size=len(X), seed=42)\n",
    "        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),\n",
    "                     num_parallel_calls=tf.data.AUTOTUNE)\n",
    "    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)\n",
    "\n",
    "train_dataset = prepare_dataset(X_train, y_train, augment=True)\n",
    "val_dataset = prepare_dataset(X_val, y_val, augment=False)\n",
    "test_dataset = prepare_dataset(X_test, y_test, augment=False)",
])

md(["## PHASE 6 — CNN Model"])

code([
    "def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):\n",
    "    model = keras.Sequential([\n",
    "        layers.Input(shape=input_shape),\n",
    "        layers.Conv2D(32, (3, 3), padding=\"same\"),\n",
    "        layers.BatchNormalization(), layers.Activation(\"relu\"),\n",
    "        layers.Conv2D(32, (3, 3), padding=\"same\"),\n",
    "        layers.BatchNormalization(), layers.Activation(\"relu\"),\n",
    "        layers.MaxPooling2D((2, 2)), layers.Dropout(0.2),\n",
    "        layers.Conv2D(64, (3, 3), padding=\"same\"),\n",
    "        layers.BatchNormalization(), layers.Activation(\"relu\"),\n",
    "        layers.Conv2D(64, (3, 3), padding=\"same\"),\n",
    "        layers.BatchNormalization(), layers.Activation(\"relu\"),\n",
    "        layers.MaxPooling2D((2, 2)), layers.Dropout(0.2),\n",
    "        layers.Conv2D(128, (3, 3), padding=\"same\"),\n",
    "        layers.BatchNormalization(), layers.Activation(\"relu\"),\n",
    "        layers.Conv2D(128, (3, 3), padding=\"same\"),\n",
    "        layers.BatchNormalization(), layers.Activation(\"relu\"),\n",
    "        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),\n",
    "        layers.Conv2D(256, (3, 3), padding=\"same\"),\n",
    "        layers.BatchNormalization(), layers.Activation(\"relu\"),\n",
    "        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),\n",
    "        layers.GlobalAveragePooling2D(),\n",
    "        layers.Dense(256), layers.BatchNormalization(), layers.Activation(\"relu\"), layers.Dropout(0.4),\n",
    "        layers.Dense(128), layers.BatchNormalization(), layers.Activation(\"relu\"), layers.Dropout(0.3),\n",
    "        layers.Dense(num_classes, activation=\"softmax\"),\n",
    "    ])\n",
    "    return model\n",
    "\n",
    "model = build_cnn()\n",
    "model.summary()",
])

md(["## PHASE 7 — Compile and Train"])

code([
    "model.compile(\n",
    "    optimizer=keras.optimizers.Adam(learning_rate=0.001),\n",
    "    loss=\"sparse_categorical_crossentropy\",\n",
    "    metrics=[\"accuracy\"],\n",
    ")\n",
    "callbacks = [\n",
    "    keras.callbacks.EarlyStopping(monitor=\"val_accuracy\", patience=20, restore_best_weights=True, verbose=1),\n",
    "    keras.callbacks.ReduceLROnPlateau(monitor=\"val_loss\", factor=0.5, patience=7, min_lr=1e-6, verbose=1),\n",
    "]\n",
    "history = model.fit(train_dataset, validation_data=val_dataset, epochs=80, callbacks=callbacks)\n",
    "print(f\"\\nCompleted after {len(history.history['loss'])} epochs.\")",
])

md(["## PHASE 8 — Evaluation"])

code([
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
    "ep = range(1, len(history.history[\"accuracy\"]) + 1)\n",
    "ax1.plot(ep, history.history[\"accuracy\"], \"b-\", label=\"Train\")\n",
    "ax1.plot(ep, history.history[\"val_accuracy\"], \"r-\", label=\"Val\")\n",
    "ax1.set_title(\"Accuracy\"); ax1.legend(); ax1.grid(True, alpha=0.3)\n",
    "ax2.plot(ep, history.history[\"loss\"], \"b-\", label=\"Train\")\n",
    "ax2.plot(ep, history.history[\"val_loss\"], \"r-\", label=\"Val\")\n",
    "ax2.set_title(\"Loss\"); ax2.legend(); ax2.grid(True, alpha=0.3)\n",
    "plt.suptitle(\"Training History\")\n",
    "plt.tight_layout()\n",
    "plt.show()",
])

code([
    "test_loss, test_acc = model.evaluate(test_dataset, verbose=0)\n",
    "print(f\"Test Accuracy: {test_acc:.4f}\")\n",
    "print(f\"Test Loss:     {test_loss:.4f}\")\n",
    "print(f\"Best val acc:  {max(history.history['val_accuracy']):.4f}\")",
])

code([
    "y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)\n",
    "print(classification_report(y_test, y_pred))",
])

code([
    "cm = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", xticklabels=range(10), yticklabels=range(10))\n",
    "plt.xlabel(\"Predicted\"); plt.ylabel(\"Actual\"); plt.title(\"Confusion Matrix\")\n",
    "plt.tight_layout()\n",
    "plt.show()",
])

md(["## PHASE 9 — Upload and Auto-Check"])

code([
    "import ipywidgets as widgets\n",
    "from IPython.display import display\n",
    "\n",
    "def auto_check(file_path):\n",
    "    img_array = preprocess_image(file_path)\n",
    "    img_array = np.expand_dims(img_array, axis=(0, -1))\n",
    "    prediction = model.predict(img_array, verbose=0)\n",
    "    predicted_digit = np.argmax(prediction)\n",
    "    confidence = np.max(prediction) * 100\n",
    "    fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n",
    "    axes[0].imshow(Image.open(file_path), cmap=\"gray\")\n",
    "    axes[0].set_title(\"Original\"); axes[0].axis(\"off\")\n",
    "    axes[1].imshow(img_array[0].squeeze(), cmap=\"gray\")\n",
    "    axes[1].set_title(\"Preprocessed\"); axes[1].axis(\"off\")\n",
    "    probs = prediction[0]\n",
    "    colors = ['green' if i == predicted_digit else 'steelblue' for i in range(10)]\n",
    "    axes[2].barh(range(10), probs, color=colors)\n",
    "    axes[2].set_yticks(range(10))\n",
    "    axes[2].set_title(\"Probabilities\")\n",
    "    plt.suptitle(f\"Predicted: Digit {predicted_digit} ({confidence:.1f}%)\", fontsize=14, fontweight=\"bold\")\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    for i, prob in enumerate(probs):\n",
    "        bar = \"#\" * int(prob * 30)\n",
    "        mark = \" <--\" if i == predicted_digit else \"\"\n",
    "        print(f\"  {i}: {prob*100:5.1f}% {bar}{mark}\")\n",
    "\n",
    "upload = widgets.FileUpload(accept='image/*', multiple=False)\n",
    "def on_upload(change):\n",
    "    if change['new']:\n",
    "        for name, info in change['new'].items():\n",
    "            with open(f\"temp_{name}\", 'wb') as f:\n",
    "                f.write(info['content'])\n",
    "            auto_check(f\"temp_{name}\")\n",
    "            os.remove(f\"temp_{name}\")\n",
    "upload.observe(on_upload, names='value')\n",
    "display(upload)\n",
    "print(\"Upload any handwritten digit image to predict!\")",
])

code([
    "# Quick test\n",
    "auto_check(\"test_digit.jpg\")",
])

md(["## PHASE 10 — Save and Deploy"])

code([
    "model.save(\"handwritten_digit_cnn.keras\")\n",
    "loaded = keras.models.load_model(\"handwritten_digit_cnn.keras\")\n",
    "_, l_acc = loaded.evaluate(test_dataset, verbose=0)\n",
    "print(f\"Model saved. Reloaded accuracy: {l_acc:.4f}\")",
])

code([
    "%%writefile app.py\n",
    "import streamlit as st\n",
    "import numpy as np\n",
    "from PIL import Image, ImageFilter\n",
    "import tensorflow as tf\n",
    "\n",
    "@st.cache_resource\n",
    "def load_model():\n",
    "    return tf.keras.models.load_model(\"handwritten_digit_cnn.keras\")\n",
    "\n",
    "model = load_model()\n",
    "\n",
    "def otsu_threshold(arr):\n",
    "    hist = np.bincount(arr.astype(np.uint8).ravel(), minlength=256)\n",
    "    total = arr.size\n",
    "    sum_total = np.sum(np.arange(256) * hist)\n",
    "    sum_bg, w_bg, max_var, thresh = 0.0, 0, 0, 0\n",
    "    for t in range(256):\n",
    "        w_bg += hist[t]\n",
    "        if w_bg == 0: continue\n",
    "        w_fg = total - w_bg\n",
    "        if w_fg == 0: break\n",
    "        sum_bg += t * hist[t]\n",
    "        var = w_bg * w_fg * (sum_bg/w_bg - (sum_total-sum_bg)/w_fg) ** 2\n",
    "        if var > max_var: max_var, thresh = var, t\n",
    "    return thresh\n",
    "\n",
    "def preprocess_image(img):\n",
    "    img = img.convert(\"L\").filter(ImageFilter.GaussianBlur(radius=1))\n",
    "    arr = np.array(img)\n",
    "    arr = np.where(arr > otsu_threshold(arr), 255, 0).astype(np.uint8)\n",
    "    if arr.mean() < 128: arr = 255 - arr\n",
    "    img = Image.fromarray(arr).resize((32, 32), Image.LANCZOS)\n",
    "    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=(0, -1))\n",
    "\n",
    "st.title(\"Handwritten Digit Recognition\")\n",
    "uploaded = st.file_uploader(\"Upload digit image\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "if uploaded:\n",
    "    image = Image.open(uploaded)\n",
    "    st.image(image, width=200)\n",
    "    if st.button(\"Predict\"):\n",
    "        pred = model.predict(preprocess_image(image))\n",
    "        digit = int(np.argmax(pred))\n",
    "        conf = float(np.max(pred)) * 100\n",
    "        st.success(f\"Digit: **{digit}** ({conf:.1f}% confidence)\")\n",
    "        for i, p in enumerate(pred[0]):\n",
    "            st.progress(float(p), text=f\"{i}: {p*100:.1f}%\")",
])

# Launch Streamlit directly from notebook
md(["## PHASE 11 — Launch Streamlit App"])

code([
    "import subprocess, time, sys, webbrowser\n",
    "\n",
    "print(\"Starting Streamlit app...\")\n",
    "print(\"A browser window will open at http://localhost:8501\")\n",
    "print(\"Close the browser tab to stop.\")\n",
    "\n",
    "# Kill any existing streamlit on port 8501\n",
    "import socket\n",
    "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n",
    "try:\n",
    "    s.connect(('localhost', 8501))\n",
    "    s.close()\n",
    "    print(\"Port 8501 already in use. Trying 8502...\")\n",
    "    PORT = 8502\n",
    "except:\n",
    "    PORT = 8501\n",
    "finally:\n",
    "    s.close()\n",
    "\n",
    "# Launch Streamlit in background\n",
    "process = subprocess.Popen(\n",
    "    [sys.executable, \"-m\", \"streamlit\", \"run\", \"app.py\", \"--server.port\", str(PORT)],\n",
    "    stdout=subprocess.PIPE,\n",
    "    stderr=subprocess.PIPE,\n",
    ")\n",
    "\n",
    "time.sleep(3)\n",
    "url = f\"http://localhost:{PORT}\"\n",
    "print(f\"\\nStreamlit is running at: {url}\")\n",
    "webbrowser.open(url)\n",
    "print(f\"Process ID: {process.pid}\")\n",
    "print(f\"To stop: run 'process.terminate()' in a new cell\")",
])

code([
    "# Stop Streamlit (run this cell to stop the app)\n",
    "try:\n",
    "    process.terminate()\n",
    "    print(\"Streamlit stopped.\")\n",
    "except:\n",
    "    print(\"No running process to stop.\")",
])

# Build notebook
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.9"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("digit_recognition.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook created: {len(cells)} cells")

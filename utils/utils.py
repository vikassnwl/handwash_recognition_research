import matplotlib.pyplot as plt
import cv2


def grid_plot(X, y=None, rows=None, figsize=None):
    rows = rows or int(np.ceil(np.sqrt(len(X))))
    cols = int(np.ceil(len(X)/rows))
    figsize = figsize or (cols*2, rows*2)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i in range(len(X))[:rows*cols]:
        if rows == 1 or cols == 1:
            ax = axes[i]
        else:
            ax = axes[i//cols][i%cols]
        if y: ax.set_title(y[i])
        ax.imshow(X[i])
    plt.tight_layout()



import tensorflow as tf
import numpy as np

def vertical_shear_fill(image, shear_intensity=0.2):
    """
    Apply vertical shear to an image and fill empty spaces with the nearest pixel values.

    Args:
        image: Input image as a NumPy array or TensorFlow tensor (H, W, C).
        shear_intensity: Shear intensity (positive or negative float).

    Returns:
        Vertically sheared image as a TensorFlow tensor.
    """
    # Get the image dimensions
    height, width, _ = image.shape

    # Define the affine transformation matrix for vertical shear
    transformation_matrix = [
        1.0, 0.0, 0.0,  # First row: [a, b, c]
        shear_intensity, 1.0, 0.0,  # Second row: [d, e, f]
        0.0, 0.0  # Third row is unused in 2D transforms
    ]

    # Apply the transformation
    sheared_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, axis=0),  # Add batch dimension
        transforms=tf.constant([transformation_matrix], dtype=tf.float32),  # Add batch dimension
        output_shape=[height, width],
        fill_value=0.0,  # Fill empty areas with black
        interpolation='BILINEAR'  # Use bilinear interpolation
    )

    # Convert to NumPy array for further processing
    sheared_image = tf.squeeze(sheared_image).numpy()

    # Fill black areas with nearest pixel values
    mask = (sheared_image == 0)  # Identify black pixels
    filled_image = np.copy(sheared_image)
    for i in range(height):
        for j in range(width):
            if np.all(mask[i, j]):  # If the pixel is black
                filled_image[i, j] = filled_image[max(i - 1, 0), j]  # Copy nearest pixel

    return tf.convert_to_tensor(filled_image, dtype=tf.float32)


def preprocess_n_predict(img_pth):
    img = cv2.imread(img_pth)[..., ::-1]
    # img = cv2.resize(img, (224, 224))
    # img = tf.keras.applications.xception.preprocess_input(img)
    img = cv2.resize(img, (256, 256))
    img = img/255.
    img = np.expand_dims(img, axis=0)
    probas = model.predict(img, verbose=0)
    label = probas.argmax()
    # return labels_dict[label]
    return img[0], label


def predict_n_plot(img_dir_pth, num_preds_to_make=None, num_rows_to_display=None, figsize=None):
    img_names = os.listdir(img_dir_pth)
    num_preds_to_make = num_preds_to_make or len(img_names)
    num_rows_to_display = num_rows_to_display or int(np.ceil(np.sqrt(num_preds_to_make)))
    num_cols_to_display = int(np.ceil(num_preds_to_make/num_rows_to_display))
    figsize = figsize or (num_cols_to_display*2, num_rows_to_display*2)
    fig, axes = plt.subplots(num_rows_to_display, num_cols_to_display, figsize=figsize)
    for i, img_name in enumerate(img_names[:num_preds_to_make]):
        img, lbl = preprocess_n_predict(f"{img_dir_pth}/{img_name}")
        if num_rows_to_display == 1 or num_cols_to_display == 1:
            ax = axes[i]
        else:
            ax = axes[i//num_cols_to_display][i%num_cols_to_display]
        ax.set_title(lbl)
        ax.imshow(img)
    plt.tight_layout()



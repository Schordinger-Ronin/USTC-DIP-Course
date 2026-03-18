import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 4, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 4, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 2)

    return marked_image

# Point-guided image deformation using Affine Moving Least Squares (MLS)
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image using Affine MLS.
    """
    if len(source_pts) < 3:
        print("Please select at least 3 point pairs for a stable Affine MLS transformation.")
        return image

    # Convert to NumPy array
    image = np.array(image)
    h, w = image.shape[:2]

    # For backward mapping, we calculate where each output pixel came from.
    # Therefore, p (control points) = target_pts, and q (deformed positions) = source_pts
    p = target_pts.astype(np.float32)
    q = source_pts.astype(np.float32)

    # Create a grid representing all pixel coordinates in the output image
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    v = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)  # Shape: (h, w, 2)

    # 1. Calculate weights w_i = 1 / |p_i - v|^{2*alpha}
    # dist2 shape will be (h, w, N)
    dist2 = np.sum((v[:, :, None, :] - p[None, None, :, :]) ** 2, axis=-1)
    dist2[dist2 < eps] = eps  # Prevent division by zero
    w_i = 1.0 / (dist2 ** alpha)
    w_sum = np.sum(w_i, axis=-1, keepdims=True)

    # 2. Calculate weighted centroids p_* and q_* -> Shape: (h, w, 2)
    p_star = np.sum(w_i[..., None] * p[None, None, :, :], axis=2) / w_sum
    q_star = np.sum(w_i[..., None] * q[None, None, :, :], axis=2) / w_sum

    # 3. Calculate relative coordinates \hat{p}_i and \hat{q}_i -> Shape: (h, w, N, 2)
    p_hat = p[None, None, :, :] - p_star[:, :, None, :]
    q_hat = q[None, None, :, :] - q_star[:, :, None, :]

    # 4. Construct the matrices for Affine MLS
    # M = (sum w_i \hat{p}_i^T \hat{p}_i)^{-1} * (sum w_i \hat{p}_i^T \hat{q}_i)
    w_phat_x = w_i * p_hat[..., 0]
    w_phat_y = w_i * p_hat[..., 1]
    
    # Elements of the 2x2 symmetric matrix (sum w_i \hat{p}_i^T \hat{p}_i)
    A = np.sum(w_phat_x * p_hat[..., 0], axis=-1)
    B = np.sum(w_phat_x * p_hat[..., 1], axis=-1)
    C = np.sum(w_phat_y * p_hat[..., 1], axis=-1)
    
    # Determinant of the 2x2 matrix
    det = A * C - B * B + eps
    
    # Inverse matrix elements
    inv_A = C / det
    inv_B = -B / det
    inv_C = A / det
    
    # Elements of the 2x2 matrix (sum w_i \hat{p}_i^T \hat{q}_i)
    R11 = np.sum(w_phat_x * q_hat[..., 0], axis=-1)
    R12 = np.sum(w_phat_x * q_hat[..., 1], axis=-1)
    R21 = np.sum(w_phat_y * q_hat[..., 0], axis=-1)
    R22 = np.sum(w_phat_y * q_hat[..., 1], axis=-1)
    
    # Compute composite Transformation Matrix M = Inverse * R
    M11 = inv_A * R11 + inv_B * R21
    M12 = inv_A * R12 + inv_B * R22
    M21 = inv_B * R11 + inv_C * R21
    M22 = inv_B * R12 + inv_C * R22
    
    # 5. Apply the deformation function f(v) = (v - p_*) * M + q_*
    v_p_x = v[..., 0] - p_star[..., 0]
    v_p_y = v[..., 1] - p_star[..., 1]
    
    f_x = v_p_x * M11 + v_p_y * M21 + q_star[..., 0]
    f_y = v_p_x * M12 + v_p_y * M22 + q_star[..., 1]
    
    # 6. Remap the image using cv2.remap for rapid backward sampling
    map_x = f_x.astype(np.float32)
    map_y = f_y.astype(np.float32)
    
    # Using BORDER_REFLECT_101 to gracefully handle pixels that get pulled in from outside original bounds
    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return warped_image

def run_warping():
    global points_src, points_dst, image
    
    if len(points_src) > 0 and len(points_src) == len(points_dst):
        warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))
        return warped_image
    return image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Point-Based Image Deformation (Moving Least Squares)")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True)
            point_select = gr.Image(label="Click to Select Source and Target Points (Blue -> Red)", interactive=True)

        with gr.Column():
            result_image = gr.Image(label="Warped Result")

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()
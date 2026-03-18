import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):
    if image is None:
        return None

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image 
    
    # 1. Translate the image center to the origin (0, 0)
    T_to_origin = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 2. Rotate and Scale (Similarity Transform)
    theta = np.radians(rotation)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    R_S = np.array([
        [scale * cos_t, scale * -sin_t, 0],
        [scale * sin_t, scale * cos_t,  0],
        [0,             0,              1]
    ], dtype=np.float32)
    
    # 3. Horizontal Flip
    flip_factor = -1 if flip_horizontal else 1
    F = np.array([
        [flip_factor, 0, 0],
        [0,           1, 0],
        [0,           0, 1]
    ], dtype=np.float32)
    
    # 4. Translate back to the image center, and apply user-defined translation (translation_x, translation_y)
    T_back_and_translate = np.array([
        [1, 0, cx + translation_x],
        [0, 1, cy + translation_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Compose these transformations in the correct order: M = T_final * F * R_S * T_origin
    # (When applying to pixel coordinates, matrix multiplication is applied from right to left)
    M_composite = T_back_and_translate @ F @ R_S @ T_to_origin
    
    # Extract the 2x3 affine matrix for OpenCV's warpAffine function
    M_affine = M_composite[:2, :]
    
    # Apply the composite transformation matrix
    transformed_image = cv2.warpAffine(image, M_affine, (w, h), borderValue=(255, 255, 255))

    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
if __name__ == "__main__":
    interactive_transform().launch()
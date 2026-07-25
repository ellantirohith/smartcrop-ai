import gradio as gr
import cv2
import numpy as np
from src.detector import SubjectDetector
from src.color_engine import ColorExtractor
from src.reframe import CanvasReframer

# Initialize modules
detector = SubjectDetector("yolov8n.pt")
color_extractor = ColorExtractor(n_colors=3)
reframer = CanvasReframer()

def process_image(input_image, aspect_ratio):
    if input_image is None:
        return None
    
    # Convert PIL Image input to OpenCV BGR array
    image_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    # Run detection, color extraction, and canvas reframing
    subject_box = detector.detect_primary_subject(image_bgr)
    bg_color = color_extractor.get_dominant_colors(image_bgr)
    output_pil = reframer.reframe(image_bgr, subject_box, bg_color, aspect_ratio)

    return output_pil

# Launch local Gradio application
with gr.Blocks(title="SmartCrop-AI | Saliency-Aware Auto-Cropper") as demo:
    gr.Markdown("# 🎨 SmartCrop-AI: Saliency-Aware Auto-Cropper & Brand Canvas Generator")
    gr.Markdown("Detects primary subjects using YOLOv8, extracts background colors via K-Means, and reframes assets across aspect ratios without cutting off content.")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload Source Image")
            ratio_dropdown = gr.Dropdown(
                choices=["1:1 (Square / Post)", "16:9 (Landscape / Banner)", "9:16 (Portrait / Story)", "4:5 (Social Feed)"],
                value="1:1 (Square / Post)",
                label="Target Aspect Ratio"
            )
            btn = gr.Button("Generate Smart Canvas", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(type="pil", label="Smart Cropped Output")

    btn.click(fn=process_image, inputs=[input_img, ratio_dropdown], outputs=output_img)

if __name__ == "__main__":
    demo.launch()
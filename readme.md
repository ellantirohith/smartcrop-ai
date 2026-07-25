# 🎨 SmartCrop-AI: Saliency-Aware Auto-Cropper & Brand Canvas Generator

> An end-to-end Computer Vision engine that detects primary subjects, extracts dominant background color palettes, and reframes graphics across social media aspect ratios without cropping out critical visual content.

---

## 📌 Features

* **Saliency-Aware Object Detection:** Identifies primary subjects using YOLOv8 to guarantee zero focal-point clipping.
* **Color Palette Extraction:** Runs K-Means Clustering on margin pixels to identify background tones automatically.
* **Dynamic Canvas Adapter:** Reframes graphics to standard formats (`1:1`, `16:9`, `9:16`, `4:5`) with color-padded canvas expansion.
* **Interactive Local UI:** Includes a single-click interactive web app powered by Gradio.

---

## 🏗️ Technical Architecture
# gradio_frontend.py

import gradio as gr
import requests
import io

def classify_with_backend(image):
    url = "http://127.0.0.1:8000/classify"
    
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()
    
    response = requests.post(url, files={"file": ("image.png", image_bytes, "image/png")})
    if response.status_code == 200:
        result = response.json()
        return result.get("label", "예측 실패")
    else:
        return f"Error: {response.status_code}"


iface = gr.Interface(
    fn=classify_with_backend,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="👕 FashionMNIST 이미지 분류기",
    description="28x28 흑백 의류 이미지 (T-shirt, Trouser 등)를 넣어보세요!"
)

if __name__ == "__main__":
    iface.launch()

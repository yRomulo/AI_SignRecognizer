import json
import time
import os
from fasthtml.common import *
from core.processor import GestureProcessor
from core.utils import decode_image, encode_image

app, rt = fast_app(hdrs=(
    Link(rel='stylesheet', href=f'/assets/style.css?v={int(time.time())}'),
    Script(src=f'/assets/script.js?v={int(time.time())}'),
))
processor = GestureProcessor()

# To track FPS per connection, we'd normally use a session or context, 
# but for simplicity in this single-user-focused app, let's use a class to track it.
class FPSTracker:
    def __init__(self):
        self.prev_time = time.time()
    def update(self):
        curr = time.time()
        fps = 1 / (curr - self.prev_time) if curr > self.prev_time else 0
        self.prev_time = curr
        return int(fps)

fps_tracker = FPSTracker()

@rt("/")
def get():
    gesture_files = sorted([f for f in os.listdir("assets/images/gestures") if f.endswith('.png')]) if os.path.exists("assets/images/gestures") else []
    
    gallery_items = []
    for f in gesture_files:
        name = f.replace('.png', '').replace('_', ' ').title()
        gallery_items.append(
            Div(
                Img(src=f"/assets/images/gestures/{f}", alt=name, style="width: 50px; height: 50px; object-fit: contain; filter: drop-shadow(0 2px 5px rgba(0,0,0,0.4));"),
                P(name, style="font-size: 0.75rem; color: #e2e8f0; margin-top: 8px; font-weight: 600; text-align: center;"),
                style="display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem; transition: transform 0.2s;",
                onmouseover="this.style.transform='translateY(-3px)'; this.style.borderColor='var(--primary)'; this.style.background='rgba(255,255,255,0.06)';",
                onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='rgba(255,255,255,0.08)'; this.style.background='rgba(255,255,255,0.03)';",
                title=name
            )
        )
        
    gestures_grid = Div(
        H3("Gestos Disponíveis para Teste", style="margin-bottom: 1.5rem; color: #94a3b8; font-size: 1.1rem;"),
        Div(*gallery_items, style="display: grid; grid-template-columns: repeat(auto-fill, minmax(90px, 1fr)); gap: 1rem; width: 100%;"),
        cls="info-card", style="width: 100%; border-radius: 24px; padding: 1.5rem; background: var(--surface); backdrop-filter: blur(12px); border: 1px solid var(--border);"
    )

    return Title("AI SignRecognizer"), Main(
        Header(
            H1("AI SignRecognizer"),
            P("Intelligent Hand Gesture Recognition System", cls="subtitle")
        ),
        Div(
            Div(
                Div(
                    Video(id="video", autoplay=True, playsinline=True, style="display:none"),
                    Canvas(id="canvas"),
                    Div("FPS: 0", id="fps-counter", cls="fps-badge"),
                    cls="vision-card"
                ),
                gestures_grid,
                style="display: flex; flex-direction: column; gap: 2rem; flex: 1; max-width: 100%; min-width: 640px;"
            ),
            Div(
                Div(
                    H3("Image Quality Control"),
                    Div(
                        Input(type="range", id="quality-slider", min="0.1", max="1.0", step="0.05", value="0.6"),
                        Span("60%", id="quality-value"),
                        cls="quality-control"
                    ),
                    H3("Settings"),
                    Div(
                        Label(Input(type="checkbox", id="draw-landmarks-cb", checked=True), " Draw Hand Landmarks"),
                        cls="settings-control"
                    ),
                    H3("Live Feed Data"),
                    Div(id="gesture-container"),
                    cls="info-card"
                ),
                Div(
                    H3("Detected Gesture"),
                    Div(Img(id="gesture-image"), cls="gesture-preview-box"),
                    cls="info-card"
                ),
                cls="sidebar-info"
            ),
            cls="main-content"
        ),
        cls="app-container"
    )

@app.ws("/ws")
async def ws(image: str, draw_landmarks: bool, record_gesture: str, send):
    img = decode_image(image)
    if img is not None:
        rec_gesture = record_gesture if record_gesture.strip() else None
        processed_img, labels, gesture_image = processor.process_frame(img, draw_landmarks, rec_gesture)
        fps = fps_tracker.update()
        await send(json.dumps({
            "image": encode_image(processed_img),
            "labels": labels,
            "gesture_image": gesture_image,
            "fps": fps
        }))

if __name__ == '__main__':
    # Deploy Configuration for Hugging Face Spaces (Port 7860)
    port = int(os.getenv("PORT", 7860))
    serve(port=port, host="0.0.0.0")

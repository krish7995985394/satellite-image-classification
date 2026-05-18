# inference_service.py
# FastAPI service: accepts multipart image upload -> runs your ResNet model ->
# prints predictions to server console (VS Code terminal) and returns JSON.

import os
import io
import sys
import traceback
from typing import List, Any, Dict

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from torchvision import transforms
from PIL import Image

# Import your model builder from train_eurosat.py
# (train_eurosat.build_model must be available in same folder)
from train_eurosat import build_model  # <- uses models.resnet18 weights

MODEL_SINGLE = "eurosat_resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="EuroSAT Inference Service")

# DEV CORS (allow front-end served from file:// or other host)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preprocessing: must match training transforms (resize+normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# Lazy global model + class names
MODEL = None
CLASS_NAMES: List[str] = []

def load_model_once():
    global MODEL, CLASS_NAMES
    if MODEL is not None:
        return

    # Expect checkpoint dict with keys "model_state_dict" and "class_to_idx"
    if not os.path.isfile(MODEL_SINGLE):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_SINGLE}")

    ckpt = torch.load(MODEL_SINGLE, map_location="cpu")
    # Support both dict-checkpoint and raw state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        class_to_idx = ckpt.get("class_to_idx") or ckpt.get("class_names")
    else:
        state = ckpt
        class_to_idx = None

    # Resolve class names list
    if isinstance(class_to_idx, dict):
        # class_to_idx: name->idx
        idx_to_name = {v:k for k,v in class_to_idx.items()}
        CLASS_NAMES = [idx_to_name[i] for i in range(len(idx_to_name))]
    elif isinstance(class_to_idx, (list, tuple)):
        CLASS_NAMES = list(class_to_idx)
    else:
        # fallback to EuroSAT default order
        CLASS_NAMES = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']

    num_classes = len(CLASS_NAMES)
    model = build_model(num_classes)  # returns resnet18 with fc=num_classes
    # robust load: allow partial loads
    try:
        model.load_state_dict(state)
    except Exception as e:
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_sd and state[k].shape == model_sd[k].shape}
        model_sd.update(filtered)
        model.load_state_dict(model_sd)
        print("[inference_service] partial load applied for checkpoint", flush=True)

    MODEL = model.to(DEVICE)
    MODEL.eval()
    print(f"[inference_service] Loaded model {MODEL_SINGLE} -> mode=single, classes={num_classes}", flush=True)


def predict_from_tensor(img_tensor: torch.Tensor, topk: int = 10) -> List[Dict[str, Any]]:
    """
    img_tensor: 1xC x H x W (CPU)
    returns list of {label, score} sorted by score desc (for single-label softmax)
    """
    load_model_once()
    x = img_tensor.to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(x)  # logits
        # single-label: softmax
        probs = torch.softmax(outputs, dim=1)[0].cpu().tolist()
        pairs = [{"label": CLASS_NAMES[i], "score": float(probs[i])} for i in range(len(probs))]
        pairs = sorted(pairs, key=lambda p: p["score"], reverse=True)
        return pairs[:topk]


@app.get("/", include_in_schema=False)
def index():
    # Serve an index.html if you placed one in ./static/index.html
    static_html = os.path.join("static", "index.html")
    if os.path.isfile(static_html):
        return FileResponse(static_html, media_type="text/html")
    return JSONResponse({"status":"ok","info":"Put static/index.html to serve UI"})


@app.post("/predict")
async def predict(image: UploadFile = File(...), topk: int = Form(10)):
    """
    Accepts form field 'image' (single file). Returns JSON:
      { "filename": [ {"label":"AnnualCrop","score":0.9879}, ... ] }
    Also prints prediction results to VS Code terminal (stdout).
    """
    try:
        load_model_once()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    fname = image.filename or "uploaded.jpg"
    try:
        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        x = transform(img).unsqueeze(0)  # 1xC x H x W (CPU)
        preds = predict_from_tensor(x, topk=topk)

        # Print to VS Code terminal (server stdout)
        print(f"[PREDICT] File: {fname}", flush=True)
        for rank, p in enumerate(preds[:topk], start=1):
            print(f"  {rank:02d}. {p['label']:<25s} score={p['score']:.4f}", flush=True)
        print("-" * 60, flush=True)

        return JSONResponse({fname: preds})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

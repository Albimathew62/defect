from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from inference_sdk import InferenceHTTPClient
import google.generativeai as genai
import base64, os, re, json, uvicorn, shutil, uuid

# ===== FASTAPI APP =====
app = FastAPI(title="Construction Defect Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== CONFIG =====
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="oj9cuEcYBAP6VF0c2CJI"
)

genai.configure(api_key="AIzaSyBQXtYhIBXlhEcuEAYQC7VRBYL-yiFcL-A")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ===== CLASS MAPPING =====
CLASS_MAPPING = {
    "paint_defect": {"discipline": "Construction", "trade": "Painting", "subtrade": "Interior Painting"},
    "plumbing_leak": {"discipline": "Construction", "trade": "Plumbing", "subtrade": "Interior Plumbing"},
    "crack": {"discipline": "Structural", "trade": "Concrete", "subtrade": "Wall Crack Repair"},
    "electrical_fault": {"discipline": "Electrical", "trade": "Wiring", "subtrade": "Circuit Issue"},
}

def extract_json(text: str) -> str | None:
    if not text:
        return None
    stripped = re.sub(r"^(?:```(?:json)?\s*|\s*```$)", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    m = re.search(r"\{[\s\S]*\}", stripped)
    return m.group(0) if m else None

def normalize_report(d: dict) -> dict:
    if not isinstance(d, dict):
        return {}
    management = d.get("management", {}) or {}
    next_steps = management.get("next_steps")
    if isinstance(next_steps, str):
        next_steps = [next_steps]
    elif not isinstance(next_steps, list):
        next_steps = []
    return {
        "issue_type": d.get("issue_type"),
        "short_description": d.get("short_description"),
        "detailed_description": d.get("detailed_description"),
        "discipline": d.get("discipline"),
        "trade": d.get("trade"),
        "subtrade": d.get("subtrade"),
        "management": {
            "status": management.get("status"),
            "priority": management.get("priority"),
            "assigned_to": management.get("assigned_to"),
            "next_steps": next_steps
        }
    }

# ========== ENDPOINT 1: Upload ==========
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(content={"error": "No image uploaded"}, status_code=400)

    # Unique filename
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "Upload successful", "filename": unique_name}

# ========== ENDPOINT 2: Analyze ==========
@app.get("/analyze/{filename}")
async def analyze(filename: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    # Roboflow Detection
    try:
        result = CLIENT.infer(filepath, model_id="")
    except Exception as e:
        result = {"error": str(e)}

    detections = result.get("predictions", [])

    if detections:
        det = detections[0]
        det_class = det.get("class", "Unknown").lower()
        mapped = CLASS_MAPPING.get(det_class, {"discipline": "Unknown", "trade": "Unknown", "subtrade": "Unknown"})

        llm_report = normalize_report({
            "issue_type": det_class,
            "short_description": f"Detected {det_class}",
            "detailed_description": f"Detected {det_class} with confidence {round(det.get('confidence', 0), 2)}",
            "discipline": mapped["discipline"],
            "trade": mapped["trade"],
            "subtrade": mapped["subtrade"],
            "management": {
                "status": "Pending",
                "priority": "Medium",
                "assigned_to": "Inspector",
                "next_steps": ["Review detection", "Confirm on-site"]
            }
        })
        return {"source": "Roboflow", "llm_report": llm_report}

    # Gemini fallback
    with open(filepath, "rb") as f:
        image_bytes = f.read()

    try:
        prompt = (
            "Analyze this construction image. Return ONLY valid JSON with keys: "
            "issue_type, short_description, detailed_description, discipline, trade, subtrade, "
            "management (object with keys: status, priority, assigned_to, next_steps as array of strings). "
            "Do not include any extra text or markdown fences."
        )

        gemini_response = gemini_model.generate_content([
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()}},
            prompt
        ])

        gemini_text = (gemini_response.text or "").strip()
        json_str = extract_json(gemini_text) or gemini_text  

        try:
            parsed = json.loads(json_str)
        except Exception:
            fallback = normalize_report({
                "issue_type": "Uncertain",
                "short_description": "Model returned unstructured text.",
                "detailed_description": gemini_text,
                "discipline": None,
                "trade": None,
                "subtrade": None,
                "management": {"status": None, "priority": None, "assigned_to": None, "next_steps": []}
            })
            return {"source": "model", "llm_report": fallback}

        llm_report = normalize_report(parsed)
        return {"source": "model", "llm_report": llm_report}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

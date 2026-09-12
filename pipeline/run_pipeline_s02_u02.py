"""Inference pipeline — Stage 1 (S02) + Stage 2 (U02) + Stage 3B OCR Fusion.

Stage 1 (S02): DINOv2-Large + MLP binary classifier — commercial (0) vs non-commercial (1).
Stage 2 (U02): DINOv2-Large + MLP binary classifier — healthy (0) vs unhealthy (1).
Stage 3B OCR:  PaddleOCR → keyword matching (Jaro-Winkler, threshold=0.85) → binary
               probability scoring (mirroring the OB02/FB10 best condition from Stage 3B)
               → weighted-average fusion with U02 vision probabilities.

Fusion rule (commercial images only):
  - OCR silent (n_matched == 0):  use U02 probabilities directly.
  - OCR active:  p_final = FUSION_ALPHA * p_u02 + (1-FUSION_ALPHA) * p_ocr

Runs one location folder at a time. Output CSV:
    <output_dir>/<city>_<location_folder_name>.csv

Usage:
    python run_pipeline_s02_u02.py \\
        <s02_model_path> \\
        <u02_model_path> \\
        <city> \\
        <location_folder> \\
        [--output-dir <output_dir>]

Checkpoint format (.pt):
    {
        "model_state_dict": ...,
        "input_dim": 1024,
        "class_dict": {0: "...", 1: "..."},
        "random_seed": 42,
    }
"""

import argparse
import logging
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image
from rapidfuzz.distance import JaroWinkler
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")           # disable oneDNN on Windows to avoid PIR conversion errors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("pipeline_s02_u02.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddleocr").setLevel(logging.ERROR)

try:
    from paddleocr import PaddleOCR
except ImportError:
    logger.error("PaddleOCR not found. Install with: pip install paddleocr")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Fusion configuration
# ---------------------------------------------------------------------------

FUSION_ALPHA = 0.5          # weight for vision branch: p_final = α·p_u02 + (1-α)·p_ocr
OCR_THRESHOLD = 0.85        # Jaro-Winkler similarity threshold for keyword matching
WEIGHT_SPECIFIC = 2.0       # higher weight for specific-indicator keywords (exact matches)
WEIGHT_GENERAL = 1.0

# ---------------------------------------------------------------------------
# OCR text cleaning  (mirrors experiments/stage3_ablation/cleaning.py)
# ---------------------------------------------------------------------------

NEGATIVE_WORDS: list[str] = [
    "google", "aluga", "motor", "escola", "cidade", "salao", "coxao", "colcao",
    "sala", "aceitamos", "aqui", "sp", "visa", "cartao", "cartoes",
]

_NON_ALPHA = re.compile(r"[^a-z\s]+")
_WHITESPACE = re.compile(r"\s+")


def _fold_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    t = _fold_diacritics(text).lower()
    t = _NON_ALPHA.sub(" ", t)
    return _WHITESPACE.sub(" ", t).strip()


def _is_negative(token: str, threshold: float = 0.8) -> bool:
    from rapidfuzz.distance import Levenshtein
    return any(
        Levenshtein.normalized_similarity(token, neg) >= threshold
        for neg in NEGATIVE_WORDS
    )


def tokenize_and_clean(raw_tokens: list[str], min_length: int = 3) -> list[str]:
    cleaned: list[str] = []
    for raw in raw_tokens:
        for word in normalize_text(raw).split():
            if len(word) < min_length:
                continue
            if _is_negative(word):
                continue
            cleaned.append(word)
    return cleaned


# ---------------------------------------------------------------------------
# Keyword matching  (mirrors experiments/stage3_ablation/matching.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Match:
    token: str
    hit: str
    similarity: float
    ocr_conf: float


def match_tokens(
    tokens_with_conf: list[tuple[str, float]],
    vocabulary: list[str],
    threshold: float = OCR_THRESHOLD,
) -> list[Match]:
    """Jaro-Winkler best-hit matching per token."""
    matches: list[Match] = []
    for token, ocr_conf in tokens_with_conf:
        best_hit, best_sim = "", 0.0
        for vocab_word in vocabulary:
            s = JaroWinkler.normalized_similarity(token, vocab_word)
            if s > best_sim:
                best_sim = s
                best_hit = vocab_word
        if best_sim >= threshold:
            matches.append(Match(token=token, hit=best_hit, similarity=best_sim, ocr_conf=ocr_conf))
    return matches


# ---------------------------------------------------------------------------
# Binary keywords  (collapsed from experiments/stage3_ablation/keywords_11class.py
# via the g0/g1g2 → healthy, g3/g4 → unhealthy mapping)
# ---------------------------------------------------------------------------

KEYWORDS_BINARY: dict[str, list[str]] = {
    "healthy": sorted(set([
        # g0: produce-market
        "milho", "mexirica", "laranja", "rucula", "verduras", "carne", "alho",
        "frango", "agriao", "economia", "quitanda", "mercearia", "pao",
        "frutas", "hortifruti", "ofertas", "verde", "sacolao", "verdurao",
        "acougue", "abobora", "mandioca", "roca", "batata", "cebolao", "kiwi",
        "mercado", "oferta", "preco", "alface", "cereais", "banana", "caqui",
        "chuchu", "tomate", "cenoura", "granja", "legumes", "maracuja",
        "limao", "abacate", "mostarda",
        # g0: butchery-store
        "peixaria", "coxa", "galo", "suina", "carnes", "file",
        "sobrecoxa", "moela", "bife", "agougue", "tulipa", "salsicha",
        # g0: restaurant
        "refeicoes", "marmitas", "salgados", "suco", "delivery", "almoco",
        "cantinho", "marmitex", "restaurante", "pizza", "pizzaria",
        # g1g2: grocery-store
        "peixes", "linguica", "maca", "supermercado", "peixe",
        "paes", "peito", "macarrao", "queijos", "biscoito",
        "doces", "mercadinho", "salgadinho", "queijo", "biscoutos",
        # g1g2: informal-settlements-market
        "bar",
        # g1g2: street-vendor
        "barraca", "pastelaria",
    ])),
    "unhealthy": sorted(set([
        # g3: bakery
        "padaria", "bolos", "bolo", "sopas", "panificadora",
        "doce", "trigo", "confeitaria",
        # g3: convenience-store
        "batatas", "picoles", "crepe", "doceria", "cremes", "cuscu", "tapioca",
        "sorvetes", "caldo", "atacado", "lanchonete", "acai", "milkshake",
        "pastei", "sorveteria", "sorvete", "sabor",
        # g4: supermarket
        "arroz", "bovino", "leite", "feijao", "figado",
        "costelinha", "pague",
        # g4: beverages
        "espetinhos", "bebidas", "carvao", "gelo", "cerveja", "chopp",
        "refrigerante", "gelada", "geladas", "copo", "cachaca", "whisky",
        "destilados", "brahma", "skol", "antarctica", "itaipava", "nova schin",
        "kaiser", "stella artois", "heineken", "original", "bohemia",
        "devassa", "glacial", "colorado", "eisenbahn", "baden baden",
        "distribuidora",
    ])),
}

SPECIFIC_INDICATORS_BINARY: dict[str, list[str]] = {
    "healthy": sorted(set([
        # g0 specific
        "acougue", "agougue", "carnes", "peixaria",  # butchery
        "hortifruti", "sacolao", "verdurao", "quitanda",  # produce-market
        "restaurante", "pizzaria",                        # restaurant
        # g1g2 specific
        "mercearia", "mercadinho",                        # grocery / informal
        "barraca", "pastelaria",                          # street-vendor
    ])),
    "unhealthy": sorted(set([
        # g3 specific
        "padaria", "panificadora", "confeitaria",    # bakery
        "lanchonete", "doceria", "sorveteria",       # convenience-store
        # g4 specific
        "supermercado", "mercado",                   # supermarket
        "bar", "cerveja", "bebidas", "chopp", "distribuidora",  # beverages
    ])),
}

# Full vocabulary (union of both binary classes — used for matching)
VOCABULARY: list[str] = sorted(
    set(KEYWORDS_BINARY["healthy"]) | set(KEYWORDS_BINARY["unhealthy"])
)

# ---------------------------------------------------------------------------
# Binary OCR scoring  (mirrors experiments/stage3b_ablation/scoring.py
# adapted for two classes; no fallback — silent → defer to vision branch)
# ---------------------------------------------------------------------------

CLASSES_BINARY = ["healthy", "unhealthy"]


def score_image_binary(
    matches: list[Match],
    temperature: float = 1.0,
) -> tuple[np.ndarray, int, str]:
    """Compute binary healthy/unhealthy probabilities from OCR keyword matches.

    Returns
    -------
    probs : np.ndarray, shape (2,)  — [p_healthy, p_unhealthy]
    n_matched : int
    decision_source : "ocr_match" | "ocr_silent_fallback"
    """
    scores = {"healthy": 0.0, "unhealthy": 0.0}
    counts = {"healthy": 0,   "unhealthy": 0}

    for m in matches:
        for cls in CLASSES_BINARY:
            if m.hit in KEYWORDS_BINARY.get(cls, []):
                is_specific = (
                    m.hit in SPECIFIC_INDICATORS_BINARY.get(cls, [])
                    and m.similarity >= 1.0
                )
                weight = WEIGHT_SPECIFIC if is_specific else WEIGHT_GENERAL
                scores[cls] += m.ocr_conf * m.similarity * weight
                counts[cls] += 1

    n_matched = sum(counts.values())

    if n_matched == 0:
        # Silent — caller uses vision branch probs directly
        return np.array([0.5, 0.5]), 0, "ocr_silent_fallback"

    # Per-class normalization, then softmax
    for cls in CLASSES_BINARY:
        if counts[cls] > 0:
            scores[cls] /= counts[cls]

    raw = np.array([scores[cls] for cls in CLASSES_BINARY], dtype=np.float64)
    probs = softmax(raw / max(temperature, 1e-6))
    return probs, n_matched, "ocr_match"


# ---------------------------------------------------------------------------
# PaddleOCR helper  (mirrors experiments/stage3_ablation/engine.py for 3.x API)
# ---------------------------------------------------------------------------

def run_ocr_on_array(ocr_engine, image_arr: np.ndarray) -> list[dict]:
    """Run PaddleOCR 3.x predict() on a numpy array.

    Returns a list of dicts with keys: text, confidence, bbox.
    """
    regions = []
    try:
        results = ocr_engine.predict(image_arr)
        if not results:
            return regions
        for page in results:
            texts  = page.get("rec_texts")  or []
            scores = page.get("rec_scores") or []
            polys  = page.get("rec_polys")  or []
            for txt, scr, poly in zip(texts, scores, polys):
                try:
                    bbox = [[float(x), float(y)] for x, y in
                            (poly.tolist() if hasattr(poly, "tolist") else poly)]
                except Exception:
                    bbox = []
                regions.append({"text": str(txt), "confidence": float(scr), "bbox": bbox})
    except Exception as exc:
        logger.warning("OCR error: %s", exc)
    return regions


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def panorama_to_perspective_fast(
    panorama: np.ndarray,
    fov_h: int = 90,
    fov_v: int = 65,
    heading: int = 0,
    pitch: int = 0,
    width: int = 800,
    height: int | None = None,
) -> np.ndarray:
    h_pano, w_pano = panorama.shape[:2]
    if height is None:
        height = int(
            width
            * math.tan(math.radians(fov_v / 2))
            / math.tan(math.radians(fov_h / 2))
            * 2
        )
    heading_rad = math.radians(heading)
    pitch_rad   = math.radians(pitch)
    fov_h_rad   = math.radians(fov_h)
    fov_v_rad   = math.radians(fov_v)

    y, x = np.indices((height, width))
    theta = (x / width  - 0.5) * fov_h_rad
    phi   = (y / height - 0.5) * fov_v_rad

    x_ray = np.cos(phi) * np.sin(theta)
    y_ray = np.sin(phi)
    z_ray = np.cos(phi) * np.cos(theta)

    y_rot   = y_ray * math.cos(pitch_rad) - z_ray * math.sin(pitch_rad)
    z_rot   = y_ray * math.sin(pitch_rad) + z_ray * math.cos(pitch_rad)
    x_final = x_ray * math.cos(heading_rad) + z_rot * math.sin(heading_rad)
    z_final = -x_ray * math.sin(heading_rad) + z_rot * math.cos(heading_rad)

    theta_final = np.arctan2(x_final, z_final)
    phi_final   = np.arcsin(np.clip(y_rot, -1, 1))

    u = (theta_final + math.pi) / (2 * math.pi)
    v = (phi_final + math.pi / 2) / math.pi

    map_x = (u * w_pano).astype(np.float32)
    map_y = (v * h_pano).astype(np.float32)
    return cv2.remap(panorama, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def list_images(folder: Path, exts: tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


# ---------------------------------------------------------------------------
# Model definitions  (S02 uses BatchNorm; U02 does not)
# ---------------------------------------------------------------------------

class _DinoMLPS02(nn.Module):
    """Stage 1 architecture: Linear → BatchNorm1d → ReLU → Dropout(0.3) → Linear."""

    def __init__(self, input_dim: int = 1024, num_classes: int = 2):
        super().__init__()
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DinoMLPU02(nn.Module):
    """Stage 2B architecture: Linear → ReLU → Dropout(0.2) → Linear."""

    def __init__(self, input_dim: int = 1024, num_classes: int = 2):
        super().__init__()
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_dict = ckpt.get("class_dict", {})
    input_dim  = ckpt.get("input_dim", 1024)
    num_classes = len(class_dict) if class_dict else ckpt.get("num_classes", 2)

    # Auto-detect architecture by checkpoint keys
    keys = list(ckpt["model_state_dict"].keys())
    has_batchnorm = any("running_mean" in k for k in keys)
    if has_batchnorm:
        model: nn.Module = _DinoMLPS02(input_dim=input_dim, num_classes=num_classes)
        arch = "S02 (BatchNorm)"
    else:
        model = _DinoMLPU02(input_dim=input_dim, num_classes=num_classes)
        arch = "U02 (no BatchNorm)"

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    torch.manual_seed(ckpt.get("random_seed", 42))
    logger.info(
        "Loaded %s — arch=%s, input_dim=%d, classes=%s",
        Path(checkpoint_path).name, arch, input_dim, class_dict,
    )
    return model


# ---------------------------------------------------------------------------
# DINOv2 feature extraction
# ---------------------------------------------------------------------------

def extract_embedding(
    image_pil: Image.Image,
    transform: T.Compose,
    backbone: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        out = backbone(pixel_values=tensor)
        return out.last_hidden_state[:, 0, :]  # (1, D)


# ---------------------------------------------------------------------------
# Per-perspective classification + fusion
# ---------------------------------------------------------------------------

def classify_perspective(
    persp_arr: np.ndarray,
    persp_pil: Image.Image,
    s02_model: nn.Module,
    u02_model: nn.Module,
    backbone: nn.Module,
    transform: T.Compose,
    ocr_engine,
    device: torch.device,
) -> dict:
    emb = extract_embedding(persp_pil, transform, backbone, device)

    # Stage 1 — S02
    with torch.no_grad():
        s02_logits = s02_model(emb)
    urban_function_class = int(s02_logits.argmax(dim=1).cpu().item())

    # defaults (non-commercial)
    vision_label        = "non_commercial"
    food_health_label   = "non_commercial"
    urban_object_class  = -1
    urban_object_probs: list[float] = []
    ocr_decision_source = "non_commercial"
    n_matched           = 0
    ocr_regions: list[dict] = []
    matched_keywords: list[str] = []
    matched_scores: list[tuple] = []

    if urban_function_class == 0:  # commercial
        # Stage 2 — U02 (vision branch)
        with torch.no_grad():
            u02_logits = u02_model(emb)
        p_u02 = u02_logits.softmax(dim=1).cpu().numpy()[0]  # [p_healthy, p_unhealthy]
        urban_object_probs = p_u02.tolist()
        vision_label = CLASSES_BINARY[int(p_u02.argmax())]
        urban_object_class = int(p_u02.argmax())

        # Stage 3B — OCR branch
        ocr_regions = run_ocr_on_array(ocr_engine, persp_arr)
        raw_texts = [r["text"] for r in ocr_regions]
        cleaned_tokens = tokenize_and_clean(raw_texts)

        # Build (token, ocr_conf) pairs using region confidence
        conf_map: dict[str, float] = {}
        for r in ocr_regions:
            for word in normalize_text(r["text"]).split():
                if len(word) >= 3:
                    conf_map.setdefault(word, r["confidence"])
        tokens_with_conf = [(t, conf_map.get(t, 0.5)) for t in cleaned_tokens]

        matches = match_tokens(tokens_with_conf, VOCABULARY)
        matched_keywords = [m.hit for m in matches]
        matched_scores   = [(m.ocr_conf, m.similarity) for m in matches]

        p_ocr, n_matched, ocr_decision_source = score_image_binary(matches)

        # Fusion
        if ocr_decision_source == "ocr_silent_fallback":
            p_final = p_u02
        else:
            p_final = FUSION_ALPHA * p_u02 + (1.0 - FUSION_ALPHA) * p_ocr

        food_health_label = CLASSES_BINARY[int(p_final.argmax())]
        urban_object_class = int(p_final.argmax())
        urban_object_probs = p_final.tolist()

    matched_kw_scores = [float(s[0]) * float(s[1]) for s in matched_scores]
    final_score = (
        sum(matched_kw_scores) / len(matched_kw_scores) if matched_kw_scores else 0.0
    )

    return {
        "urban_function_class": urban_function_class,
        "urban_object_class":   urban_object_class,
        "urban_object_probs":   urban_object_probs,
        "vision_label":         vision_label,
        "food_health_label":    food_health_label,
        "ocr_decision_source":  ocr_decision_source,
        "n_matched":            n_matched,
        "ocr_result":           ocr_regions,
        "matched_keywords":     matched_keywords,
        "scores":               matched_scores,
        "matched_keywords_scores": matched_kw_scores,
        "final_score":          final_score,
    }


# ---------------------------------------------------------------------------
# Location processing
# ---------------------------------------------------------------------------

def process_location(
    city: str,
    loc_dir: Path,
    s02_model: nn.Module,
    u02_model: nn.Module,
    backbone: nn.Module,
    transform: T.Compose,
    ocr_engine,
    device: torch.device,
) -> list[dict]:
    location = loc_dir.name
    images = list_images(loc_dir)
    if not images:
        logger.warning("No images found in %s", loc_dir)
        return []

    rows: list[dict] = []
    for img_path in tqdm(images, desc=f"[{city}/{location}]", unit="img"):
        try:
            pano_pil = Image.open(img_path).convert("RGB")
            pano_arr = np.array(pano_pil)
        except Exception as exc:
            logger.error("Cannot open %s: %s", img_path, exc)
            continue

        for heading in [0, 90, 180, 270]:
            persp_arr = panorama_to_perspective_fast(
                pano_arr, fov_h=90, heading=heading, width=1024, height=1024
            )
            persp_pil = Image.fromarray(persp_arr)

            result = classify_perspective(
                persp_arr=persp_arr,
                persp_pil=persp_pil,
                s02_model=s02_model,
                u02_model=u02_model,
                backbone=backbone,
                transform=transform,
                ocr_engine=ocr_engine,
                device=device,
            )

            rows.append({
                "image_path":       str(img_path),
                "city":             city,
                "location":         location,
                "panorama_heading": heading,
                **result,
            })

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "pipeline"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run S02+U02+OCR pipeline for a single location folder."
    )
    parser.add_argument("s02_model_path", help="Path to S02 checkpoint (.pt)")
    parser.add_argument("u02_model_path", help="Path to U02 checkpoint (.pt)")
    parser.add_argument("city", help="City name (used in output filename)")
    parser.add_argument("location_folder", help="Path to the location folder containing images")
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help=f"Directory to write CSVs (default: {_DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    loc_dir = Path(args.location_folder).resolve()
    if not loc_dir.is_dir():
        logger.error("location_folder does not exist: %s", loc_dir)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.city}_{loc_dir.name}.csv"

    if output_path.exists():
        logger.warning("Output already exists, skipping: %s", output_path)
        sys.exit(0)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    logger.info("Device: %s", device)

    logger.info("Loading DINOv2-Large backbone…")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    backbone  = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize((224, 224)),
        T.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    s02_model = load_model(args.s02_model_path, device)
    u02_model = load_model(args.u02_model_path, device)

    logger.info("Initializing PaddleOCR 3.x (lang=pt)…")
    ocr_engine = PaddleOCR(
        lang="pt",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )

    logger.info("Processing %s / %s…", args.city, loc_dir.name)
    rows = process_location(
        city=args.city,
        loc_dir=loc_dir,
        s02_model=s02_model,
        u02_model=u02_model,
        backbone=backbone,
        transform=transform,
        ocr_engine=ocr_engine,
        device=device,
    )

    if not rows:
        logger.warning("No images processed. Output CSV will be empty.")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)


if __name__ == "__main__":
    main()

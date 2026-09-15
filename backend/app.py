import os
import sys
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from skimage.metrics import structural_similarity as ssim


# =========================================================
# PATH SETUP
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


COVER_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cover"
)

UPLOAD_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "uploads"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# =========================================================
# IMPORT EXISTING PROJECT CODE
# =========================================================

from preprocessing import (
    rgb_to_ycbcr,
    ycbcr_to_rgb
)

from embedding_standard import (
    embed_standard_epvd
)

from embedding import (
    embed_ai_epvd
)

from embedding_secure import (
    embed_secure_ai_epvd
)

from extraction import (
    extract_standard_epvd
)

from extraction_ai_epvd import (
    extract_ai_epvd
)

from extraction_secure import (
    extract_secure_ai_epvd
)


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

CORS(app)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def bits_to_utf8(bits):
    usable_length = (len(bits) // 8) * 8
    bits = bits[:usable_length]

    data = bytearray()

    for i in range(0, len(bits), 8):
        data.append(int(bits[i:i + 8], 2))

    return bytes(data).decode(
        "utf-8",
        errors="replace"
    )


def text_to_bits(text):
    """
    Convert UTF-8 text into binary.
    """
    data = text.encode("utf-8")

    return "".join(
        format(byte, "08b")
        for byte in data
    )


def calculate_metrics(
    cover_rgb,
    stego_rgb,
    embedded_bits,
    total_pixels
):
    """
    Calculate image quality metrics.
    """

    cover = cover_rgb.astype(np.float64)
    stego = stego_rgb.astype(np.float64)

    mse = float(
        np.mean(
            (cover - stego) ** 2
        )
    )

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(
            10 * np.log10(
                (255 ** 2) / mse
            )
        )

    ssim_value = float(
        ssim(
            cover_rgb,
            stego_rgb,
            channel_axis=2,
            data_range=255
        )
    )

    bpp = float(
        embedded_bits / total_pixels
    )

    return {
        "mse": round(mse, 6),
        "psnr": round(psnr, 6),
        "ssim": round(ssim_value, 6),
        "bpp": round(bpp, 6)
    }


def save_stego_image(
    stego_rgb
):
    """
    Save generated stego image.
    """

    filename = (
        f"stego_"
        f"{uuid.uuid4().hex}.png"
    )

    output_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    stego_bgr = cv2.cvtColor(
        stego_rgb,
        cv2.COLOR_RGB2BGR
    )

    success = cv2.imwrite(
        output_path,
        stego_bgr
    )

    if not success:
        raise RuntimeError(
            "Failed to save stego image"
        )

    return filename


# =========================================================
# HEALTH
# =========================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "ok",
        "backend": "Flask",
        "project":
            "AI-Driven Adaptive "
            "Enhanced Pixel Value "
            "Differencing"
    })


# =========================================================
# COVERS
# =========================================================

@app.route(
    "/api/covers",
    methods=["GET"]
)
def get_covers():

    if not os.path.exists(
        COVER_DIR
    ):

        return jsonify({
            "error":
                "Cover directory not found"
        }), 404

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp"
    }

    covers = []

    for filename in sorted(
        os.listdir(COVER_DIR)
    ):

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in allowed_extensions:

            covers.append({
                "name": filename,
                "url":
                    f"/api/files/cover/"
                    f"{filename}"
            })

    return jsonify({
        "covers": covers,
        "count": len(covers)
    })


# =========================================================
# FILE SERVING
# =========================================================

@app.route(
    "/api/files/<file_type>/<filename>",
    methods=["GET"]
)
def get_file(
    file_type,
    filename
):

    directories = {
        "cover": COVER_DIR,
        "upload": UPLOAD_DIR
    }

    directory = directories.get(
        file_type
    )

    if directory is None:

        return jsonify({
            "error":
                "Invalid file type"
        }), 400

    file_path = os.path.join(
        directory,
        filename
    )

    if not os.path.isfile(
        file_path
    ):

        return jsonify({
            "error":
                "File not found"
        }), 404

    return send_from_directory(
        directory,
        filename
    )


# =========================================================
# UPLOAD COVER
# =========================================================

@app.route("/api/upload-cover", methods=["POST"])
def upload_cover():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400

        allowed = {".png", ".jpg", ".jpeg"}

        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in allowed:
            return jsonify({
                "error": "Only PNG, JPG and JPEG images are supported"
            }), 400

        # Generate a safe unique filename
        filename = f"cover_{uuid.uuid4().hex}{ext}"

        save_path = os.path.join(UPLOAD_DIR, filename)

        file.save(save_path)

        # Verify image can actually be read
        image = cv2.imread(save_path)

        if image is None:
            os.remove(save_path)
            return jsonify({"error": "Invalid image file"}), 400

        height, width = image.shape[:2]

        return jsonify({
            "success": True,
            "filename": filename,
            "image_url": f"/api/files/upload/{filename}",
            "width": width,
            "height": height
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================================================
# EMBED
# =========================================================

@app.route(
    "/api/embed",
    methods=["POST"]
)
def embed():

    start_time = time.perf_counter()

    try:

        # -------------------------------------------------
        # Request data
        # -------------------------------------------------

        req_data = request.get_json(silent=True) or request.form or request.values

        cover_filename = req_data.get(
            "cover"
        )

        message = req_data.get(
            "message"
        )

        secret_key = req_data.get(
            "secret_key",
            ""
        )

        mode = str(req_data.get(
            "mode",
            "ai"
        )).lower()

        if not cover_filename:

            return jsonify({
                "success": False,
                "error":
                    "Cover image is required"
            }), 400

        if not message:

            return jsonify({
                "success": False,
                "error":
                    "Secret message is required"
            }), 400

        allowed_modes = {
            "standard",
            "ai",
            "secure"
        }

        if mode not in allowed_modes:

            return jsonify({
                "success": False,
                "error":
                    "Allowed modes: 'standard', 'ai', 'secure'"
            }), 400

        if mode == "secure" and not secret_key:

            return jsonify({
                "success": False,
                "error":
                    "Secret key is required for secure mode"
            }), 400

        # -------------------------------------------------
        # Security: prevent path traversal
        # -------------------------------------------------

        safe_filename = os.path.basename(
            cover_filename
        )

        if safe_filename.startswith("cover_"):
            cover_path = os.path.join(
                UPLOAD_DIR,
                safe_filename
            )
        else:
            cover_path = os.path.join(
                COVER_DIR,
                safe_filename
            )

        # Fallback if image exists in other directory
        if not os.path.isfile(cover_path):
            alt_path = os.path.join(
                UPLOAD_DIR if not safe_filename.startswith("cover_") else COVER_DIR,
                safe_filename
            )
            if os.path.isfile(alt_path):
                cover_path = alt_path

        if not os.path.isfile(
            cover_path
        ):

            return jsonify({
                "success": False,
                "error":
                    "Cover image not found"
            }), 404

        # -------------------------------------------------
        # Read cover
        # -------------------------------------------------

        cover_bgr = cv2.imread(
            cover_path
        )

        if cover_bgr is None:

            return jsonify({
                "success": False,
                "error":
                    "Unable to read cover image"
            }), 400

        cover_rgb = cv2.cvtColor(
            cover_bgr,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------------------
        # Convert secret to bits
        # -------------------------------------------------

        secret_bits = text_to_bits(
            message
        )

        # -------------------------------------------------
        # STANDARD EPVD
        # -------------------------------------------------

        if mode == "standard":

            ycbcr = rgb_to_ycbcr(
                cover_rgb
            )

            y_channel = ycbcr[
                :, :, 0
            ].copy()

            (
                stego_y,
                embedded_bits,
                pairs_used
            ) = embed_standard_epvd(
                y_channel,
                secret_bits
            )

            ycbcr[
                :, :, 0
            ] = stego_y

            stego_rgb = ycbcr_to_rgb(
                ycbcr
            )

            extra = {
                "pairs_used":
                    int(pairs_used),
                "blocks_used": None
            }

        elif mode == "ai":

            temporary_filename = (
                f"temp_"
                f"{uuid.uuid4().hex}.png"
            )

            temporary_path = os.path.join(
                UPLOAD_DIR,
                temporary_filename
            )

            embed_ai_epvd(
                cover_path,
                temporary_path,
                message
            )

            stego_bgr = cv2.imread(
                temporary_path
            )

            if stego_bgr is None:

                raise RuntimeError(
                    "AI embedding failed"
                )

            stego_rgb = cv2.cvtColor(
                stego_bgr,
                cv2.COLOR_BGR2RGB
            )

            embedded_bits = len(
                secret_bits
            )

            extra = {
                "pairs_used": None,
                "blocks_used": None
            }

        else:

            # =============================================
            # SECURE AI + CHAOTIC EPVD
            # =============================================

            stego_filename = (
                f"stego_"
                f"{uuid.uuid4().hex}.png"
            )

            stego_path = os.path.join(
                UPLOAD_DIR,
                stego_filename
            )

            secure_result = embed_secure_ai_epvd(
                cover_path,
                stego_path,
                message,
                secret_key
            )

            stego_bgr = cv2.imread(
                stego_path,
                cv2.IMREAD_UNCHANGED
            )

            if stego_bgr is None:

                raise RuntimeError(
                    "Secure AI embedding failed"
                )

            stego_rgb = cv2.cvtColor(
                stego_bgr,
                cv2.COLOR_BGR2RGB
            )

            embedded_bits = int(
                secure_result["embedded_bits"]
            )

            extra = {
                "pairs_used": None,
                "blocks_used":
                    int(secure_result["blocks_used"]),
                "complete":
                    bool(secure_result["complete"])
            }

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        total_pixels = (
            cover_rgb.shape[0]
            * cover_rgb.shape[1]
        )

        metrics = calculate_metrics(
            cover_rgb,
            stego_rgb,
            embedded_bits,
            total_pixels
        )

        # -------------------------------------------------
        # Save final stego (if not already saved by secure mode)
        # -------------------------------------------------

        if mode != "secure":
            stego_filename = save_stego_image(
                stego_rgb
            )

        execution_time = (
            time.perf_counter()
            - start_time
        )

        # -------------------------------------------------
        # Response
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "mode": mode,

            "message_bits":
                len(secret_bits),

            "embedded_bits":
                int(embedded_bits),

            "remaining_bits":
                max(
                    0,
                    len(secret_bits)
                    - embedded_bits
                ),

            "capacity_utilization":
                round(
                    (
                        embedded_bits
                        / len(secret_bits)
                    ) * 100,
                    2
                ),

            "stego": stego_filename,

            "stego_image": (
                "/api/files/upload/"
                + stego_filename
            ),

            "execution_time":
                round(
                    execution_time,
                    6
                ),

            **metrics,

            **extra
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================================================
# UPLOAD STEGO
# =========================================================

@app.route("/api/upload-stego", methods=["POST"])
def upload_stego():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No stego image provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400

        ext = os.path.splitext(file.filename)[1].lower()

        if ext != ".png":
            return jsonify({
                "error": "Stego image must be a PNG file"
            }), 400

        filename = f"stego_upload_{uuid.uuid4().hex}.png"
        save_path = os.path.join(UPLOAD_DIR, filename)

        # Save the uploaded bytes directly.
        file.save(save_path)

        # Verify that OpenCV can read it.
        image = cv2.imread(
            save_path,
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            os.remove(save_path)
            return jsonify({
                "error": "Invalid PNG image"
            }), 400

        height, width = image.shape[:2]

        return jsonify({
            "success": True,
            "filename": filename,
            "image_url": f"/api/files/upload/{filename}",
            "width": width,
            "height": height
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# =========================================================
# EXTRACT (RECEIVER-ONLY)
# =========================================================

@app.route("/api/extract", methods=["POST"])
def extract_message():
    try:
        stego = request.form.get("stego") or (request.get_json(silent=True) or {}).get("stego") or request.values.get("stego")
        secret_key = request.form.get("secret_key") or (request.get_json(silent=True) or {}).get("secret_key") or request.values.get("secret_key")

        if not stego:
            return jsonify({
                "error": "Stego image is required"
            }), 400

        if not secret_key:
            return jsonify({
                "error": "Secret key is required"
            }), 400

        safe_stego = os.path.basename(stego)
        stego_path = os.path.join(
            UPLOAD_DIR,
            safe_stego
        )

        if not os.path.exists(stego_path):
            return jsonify({
                "error": f"Stego image not found: {stego}"
            }), 404

        start_time = time.time()

        (
            extracted_bits,
            extracted_message,
            blocks_used
        ) = extract_secure_ai_epvd(
            stego_path,
            secret_key
        )

        execution_time = time.time() - start_time

        return jsonify({
            "success": True,
            "extracted_message": extracted_message,
            "extracted_bits": len(extracted_bits),
            "blocks_used": blocks_used,
            "execution_time": round(
                execution_time,
                6
            )
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
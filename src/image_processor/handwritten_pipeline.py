import argparse
import base64
import json
import re
import sys
import textwrap
from pathlib import Path
import cv2
import numpy as np
import os
import requests
import tempfile
import shutil
import os
import requests
from dotenv import load_dotenv

# # OLLAMA_URL = "http://10.1.237.104:11434/api/chat"
# OLLAMA_URL  = "http://103.67.43.152/ollama2/api/chat"
# MODEL = "qwen3-vl:8b-instruct-bf16"

load_dotenv()

# ──────────────────────────────────────────────
# Config (Dynamic Provider)
# ──────────────────────────────────────────────
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "ollama").lower()

# Ollama Config
OLLAMA_URL  = os.getenv("OCR_OLLAMA_URL", "http://103.67.43.152/ollama2/api/chat")
OLLAMA_MODEL = os.getenv("OCR_OLLAMA_MODEL", "qwen3-vl:8b-instruct-bf16")

# Gemini Config
OCR_GEMINI_API_KEY = os.getenv("OCR_GEMINI_API_KEY", "")
OCR_GEMINI_MODEL = os.getenv("OCR_GEMINI_MODEL", "gemini-2.5-flash")

# OpenAI Config
OCR_OPENAI_API_KEY = os.getenv("OCR_OPENAI_API_KEY", "")
OCR_OPENAI_MODEL = os.getenv("OCR_OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 0.1
MAX_TOKENS = 6000
MAX_RETRIES = 2


def load_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _ollama_chat(messages: list, max_tokens: int) -> str:
    payload = {
        "model":    OLLAMA_MODEL,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": TEMPERATURE, "top_p": 0.9, "num_predict": max_tokens},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data    = resp.json()
    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Empty response from Ollama:\n{data}")
    return content

def _openai_chat(messages: list, max_tokens: int) -> str:
    from openai import OpenAI
    if not OCR_OPENAI_API_KEY:
        raise ValueError("OCR_OPENAI_API_KEY is not set in .env")
    
    client = OpenAI(api_key=OCR_OPENAI_API_KEY)
    formatted_messages = []
    
    for msg in messages:
        role = msg["role"]
        content_text = msg["content"]
        images = msg.get("images", [])

        if not images:
            formatted_messages.append({"role": role, "content": content_text})
        else:
            content_array = [{"type": "text", "text": content_text}]
            for img_b64 in images:
                content_array.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            formatted_messages.append({"role": role, "content": content_array})

    response = client.chat.completions.create(
        model=OCR_OPENAI_MODEL,
        messages=formatted_messages,
        # temperature=TEMPERATURE,
        max_tokens=max_tokens,
        # top_p=0.9
    )
    return response.choices[0].message.content.strip()

def _gemini_chat(messages: list, max_tokens: int) -> str:
    import google.generativeai as genai
    if not OCR_GEMINI_API_KEY:
        raise ValueError("OCR_GEMINI_API_KEY is not set in .env")
    
    genai.configure(api_key=OCR_GEMINI_API_KEY)
    
    system_instruction = None
    user_parts = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            user_parts.append(msg["content"])
            for img_b64 in msg.get("images", []):
                mime_type = "image/jpeg"
                if img_b64.startswith("iVBORw0KGgo"):
                    mime_type = "image/png"
                elif img_b64.startswith("UklGR"):
                    mime_type = "image/webp"

                user_parts.append({
                    "mime_type": mime_type,
                    "data": img_b64
                })
    
    kwargs = {"model_name": OCR_GEMINI_MODEL}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
        
    model = genai.GenerativeModel(**kwargs)
    
    # 💡 PERUBAHAN: Kita menghapus generation_config (max_tokens) 
    # agar Gemini menggunakan limit output maksimal bawaannya (8192 token)
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    response = model.generate_content(
        user_parts,
        safety_settings=safety_settings
    )
    
    try:
        return response.text.strip()
    except ValueError:
        # Fallback jika gambar masih ditolak atau model gagal mengekstrak part
        if response.candidates:
            finish_reason = response.candidates[0].finish_reason
            raise RuntimeError(f"Gemini terhenti. Finish Reason: {finish_reason}. (1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION)")
        else:
            raise RuntimeError("Respon kosong atau diblokir oleh Gemini.")

def vision_chat(messages: list, max_tokens: int = MAX_TOKENS) -> str:
    """Fungsi wrapper pendistribusi request ke model yang diplih di .env"""
    if OCR_PROVIDER == "openai":
        return _openai_chat(messages, max_tokens)
    elif OCR_PROVIDER == "gemini":
        return _gemini_chat(messages, max_tokens)
    else:
        return _ollama_chat(messages, max_tokens)


def strip_json_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def strip_html_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:html)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _tight_crop_by_content(img: np.ndarray, min_pad: int = 4) -> np.ndarray:
    if img is None or img.size == 0:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # 1. Threshold warna: ambil yang bukan putih terang (kertas biasanya > 245)
    _, mask_color = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    # 2. Canny Edge: tangkap garis pinggir warna cerah yang lolos dari threshold
    mask_edge = cv2.Canny(gray, 50, 150)

    # 3. Gabungkan mask
    mask = cv2.bitwise_or(mask_color, mask_edge)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return img  # Fallback ke asli jika blank

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    ih, iw = img.shape[:2]
    r1 = max(0, rmin - min_pad)
    r2 = min(ih, rmax + min_pad + 1)
    c1 = max(0, cmin - min_pad)
    c2 = min(iw, cmax + min_pad + 1)

    return img[r1:r2, c1:c2]


PHASE1_SYSTEM = (
    "You are an expert document and handwriting analyst specializing in OCR and digital conversion of both printed and handwritten documents. "
    "Inspect the document image and return ONLY a valid JSON object. "
    "No prose, no markdown fences, no comments."
)

PHASE1_USER = """/no_think
Carefully examine EVERY region of this document image — top to bottom, left to right.
Determine if the document is a clean digital file, a scan of a printed document, or a handwritten note.
Return a single JSON object with EXACTLY these keys.
Output ONLY the JSON. Do NOT add comments or extra text.

{
    "document_type": "memo_dinas|surat_resmi|surat_perintah|sk|invoice|form|report|table_only|mixed|handwritten_note",
    "is_handwritten": true|false,
    "handwriting_style": "cursive|block|mixed|none",
    "is_scanned_image": true|false,
    "language": "Indonesian|English|mixed",

    "header_layout": "text_only|none",
    "has_letterhead": true,
    "has_horizontal_line_below_header": true,

    "has_key_value_block": true,
    "key_value_multiline": false,
    "has_nested_key_value": false,
    "key_value_label_examples": ["Menimbang", "Mengingat"],

    "has_data_table": false,
    "has_hand_drawn_table": false,
    "table_has_merged_cells": false,

    "has_numbered_list": false,
    "has_bullet_list": false,
    "has_dash_list": false,
    "list_style": "sequential|alphabetic|roman|mixed|none",

    "has_signature_block": true,
    "signature_position": "right|left|center|dual|none",
    "has_bottom_dual_column": false,
    "has_tembusan_block": false,
    "tembusan_position": "bottom_left|bottom_right|none",

    "has_place_date_line": false,
    "place_date_position": "bottom_right|bottom_left|none",

    "has_footer": false,

    "text_alignment": "left|justify|center|mixed",
    "has_indented_paragraphs": false,

    "has_bold_text": true,
    "has_italic_text": false,
    "has_underlined_text": false,

    "heading_levels": 1,
    "font_style": "serif|sans-serif|monospace|mixed|handwritten",
    "complexity": "simple|moderate|complex"
}

Replace example values with ACTUAL values from the image.
Pay special attention to the BOTTOM SECTION of the document:
    - If there is a "Tembusan" block on the left AND a signature block on the right at the same vertical position, set has_bottom_dual_column = true.
    - If there is a "Ditetapkan di / Tanggal" line anywhere, set has_place_date_line = true.
"""


def phase1_analyse(image_b64: str) -> dict:
    print("[Phase 1] Analysing document structure ...")
    messages = [
        {"role": "system", "content": PHASE1_SYSTEM},
        {"role": "user", "content": PHASE1_USER, "images": [image_b64]},
    ]
    raw = vision_chat(messages, max_tokens=1024)
    cleaned = strip_json_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Phase 1 invalid JSON.\nRaw:\n{raw}\nCleaned:\n{cleaned}\nError:{e}"
        )


def phase15_validate(doc: dict, image_b64: str) -> dict:
    print("[Phase 1.5] Validating analysis ...")
    questions = []

    if not doc.get("has_bottom_dual_column"):
        questions.append(
            "Q1: Is there a 'Tembusan' block on the BOTTOM-LEFT of the document "
            "AND a signature/TTD block on the BOTTOM-RIGHT, at roughly the same vertical level? "
            "Answer: yes or no."
        )
    if not doc.get("has_place_date_line"):
        questions.append(
            "Q2: Is there a line like 'Ditetapkan di : ...' or 'Tanggal : ...' "
            "anywhere in the document? Answer: yes or no."
        )
    if not doc.get("key_value_multiline"):
        questions.append(
            "Q3: Does any label in the colon-aligned block (like 'Menimbang') have "
            "multiple numbered sub-items (1. 2. 3.) as its value? Answer: yes or no."
        )
    if not doc.get("has_nested_key_value"):
        questions.append(
            "Q4: Is there a sub-block of personal data fields (Nama, Jenis Kelamin, "
            "Tempat Tanggal Lahir, Jabatan, etc.) appearing indented under the main text? "
            "Answer: yes or no."
        )

    if not doc.get("table_has_merged_cells"):
        questions.append(
            "Q5: Does the data table have any merged cells? For example, a column header that spans across multiple columns (like a title over several dates), or a row header that spans multiple rows vertically. Answer: yes or no."
        )

    if not doc.get("is_handwritten"):
        questions.append(
            "Q6: Does this document contain ANY handwritten text, signatures, or annotations? Answer: yes or no."
        )

    if not questions:
        print("[Phase 1.5] No corrections needed.")
        return doc

    prompt = (
        "/no_think\n"
        "Answer these layout questions about the document image. "
        "Answer ONLY 'yes' or 'no' for each, one per line.\n\n" + "\n".join(questions)
    )
    messages = [{"role": "user", "content": prompt, "images": [image_b64]}]
    raw = vision_chat(messages, max_tokens=128)
    lines = [l.strip().lower() for l in raw.strip().splitlines() if l.strip()]

    q_map = {q[:2]: q for q in questions}
    corrections: dict = {}

    for i, line in enumerate(lines):
        is_yes = line.startswith("yes") or ": yes" in line
        q_key = f"Q{i+1}"
        q_text = questions[i] if i < len(questions) else ""
        if "Q1" in q_text and is_yes:
            corrections["has_bottom_dual_column"] = True
            corrections["has_tembusan_block"] = True
            corrections["tembusan_position"] = "bottom_left"
        elif "Q2" in q_text and is_yes:
            corrections["has_place_date_line"] = True
            corrections["place_date_position"] = "bottom_right"
        elif "Q3" in q_text and is_yes:
            corrections["key_value_multiline"] = True
        elif "Q4" in q_text and is_yes:
            corrections["has_nested_key_value"] = True
        elif "Q5" in q_text and is_yes:
            corrections["table_has_merged_cells"] = True
        elif "Q6" in q_text and is_yes:
            corrections["is_handwritten"] = True

    if corrections:
        print(f"[Phase 1.5] Corrections applied: {corrections}")
        doc.update(corrections)
    else:
        print("[Phase 1.5] No corrections needed.")
    return doc


KV_SIMPLE_EXAMPLE = """\
<table border='0' cellpadding='0' cellspacing='0' style='margin-bottom:6pt;'>
    <tr>
        <td style='padding-right:8px; vertical-align:top; white-space:nowrap;'>Kepada Yth</td>
        <td style='vertical-align:top; white-space:nowrap;'>:</td>
        <td style='padding-left:8px; vertical-align:top;'>Kepala BPKSDM</td>
    </tr>
    <tr>
        <td style='padding-right:8px; vertical-align:top; white-space:nowrap;'>Dari</td>
        <td style='vertical-align:top; white-space:nowrap;'>:</td>
        <td style='padding-left:8px; vertical-align:top;'>Direktur Jenderal</td>
    </tr>
</table>"""

KV_MULTILINE_EXAMPLE = """\
<table border='0' cellpadding='0' cellspacing='0' style='margin-bottom:6pt;'>
    <tr>
        <td style='padding-right:8px; vertical-align:top; white-space:nowrap;'>Menimbang</td>
        <td style='vertical-align:top; white-space:nowrap;'>:</td>
        <td style='padding-left:8px; vertical-align:top;'>
            1. Undang-undang RI Np. 17 Tahun 2011 tentang Intelijen Negara.<br/>
            2. Kepres No. 34 Tahun 2010 tentang Badan Intelijen Negara.<br/>
            3. Peraturan Kepala BIN Nomor. 41 Tahun 2006 tentang penyesuaian tugas pokok.
        </td>
    </tr>
    <tr>
        <td style='padding-right:8px; vertical-align:top; white-space:nowrap;'>Mengingat</td>
        <td style='vertical-align:top; white-space:nowrap;'>:</td>
        <td style='padding-left:8px; vertical-align:top;'>
            Masih adanya Pungutan liar yang di lakukan oleh oknum – oknum...
        </td>
    </tr>
    <tr>
        <td style='padding-right:8px; vertical-align:top; white-space:nowrap;'>Memperhatikan</td>
        <td style='vertical-align:top; white-space:nowrap;'>:</td>
        <td style='padding-left:8px; vertical-align:top;'>
        Perintah Panglima Tertinggi Presiden Republik Indonesia
        </td>
    </tr>
</table>"""

KV_NESTED_EXAMPLE = """\
<table border='0' cellpadding='0' cellspacing='0' style='margin-bottom:6pt;'>
    <tr>
        <td style='padding-right:8px; vertical-align:top; white-space:nowrap;'>Memutuskan</td>
        <td style='vertical-align:top; white-space:nowrap;'>:</td>
        <td style='padding-left:8px; vertical-align:top;'>
        Perlu segera dibentuk Satuan Kerja Khusus...<br/><br/>
        Terhitung Mulai Tanggal Terbitnya Surat Perintah ini Menegaskan :<br/>
            <table border='0' cellpadding='0' cellspacing='0' style='margin-left:0; margin-top:4px;'>
                <tr>
                    <td style='padding-right:8px; vertical-align:top; white-space:nowrap;'>Nama</td>
                    <td style='vertical-align:top; white-space:nowrap;'>:</td>
                    <td style='padding-left:8px; vertical-align:top;'>RACHMAT PANJI DARMA</td>
                </tr>
                <tr>
                    <td style='padding-right:8px; vertical-align:top; white-space:nowrap;'>Jenis Kelamin</td>
                    <td style='vertical-align:top; white-space:nowrap;'>:</td>
                    <td style='padding-left:8px; vertical-align:top;'>Laki-laki</td>
                </tr>
                <tr>
                    <td style='padding-right:8px; vertical-align:top; white-space:nowrap;'>Jabatan</td>
                    <td style='vertical-align:top; white-space:nowrap;'>:</td>
                    <td style='padding-left:8px; vertical-align:top;'>Agen Pembina Madya Tingkat I</td>
                </tr>
            </table>
        </td>
    </tr>
</table>"""

DUAL_BOTTOM_EXAMPLE = """\
<div style='display:flex; justify-content:space-between; align-items:flex-start; margin-top:16pt;'>
  <div style='min-width:200px;'>
    <p style='margin:0; margin-bottom:4pt;'><strong>Tembusan :</strong></p>
    <p style='margin:0; margin-bottom:2pt;'>Kepada Yth,</p>
    <p style='margin:0; margin-bottom:2pt;'>1. Presiden Republik Indonesia</p>
    <p style='margin:0; margin-bottom:2pt;'>2. Menkopolhukam</p>
    <p style='margin:0; margin-bottom:2pt;'>3. Panglima TNI</p>
  </div>
  <div style='text-align:center; min-width:220px;'>
    <table border='0' cellpadding='0' cellspacing='2' style='margin:0 auto; text-align:left;'>
      <tr>
        <td style='white-space:nowrap; padding-right:6px;'>Ditetapkan di</td>
        <td>:</td>
        <td style='padding-left:6px;'>Jakarta</td>
      </tr>
      <tr>
        <td style='white-space:nowrap; padding-right:6px;'>Tanggal</td>
        <td>:</td>
        <td style='padding-left:6px;'>16 Januari 2017</td>
      </tr>
    </table>
    <br/>
    <strong>BADAN INTELIJEN NEGARA</strong><br/>
    <strong>KEPALA SATKORLAK OPSINSUS</strong><br/>
    <strong>SEKURITAS ASET PERBENDAHARAAN NEGARA</strong><br/>
    <br/><br/><br/>
    <strong>( ERRY MARSONO )</strong>
  </div>
</div>
<div style='clear:both;'></div>"""


def build_extraction_prompt(doc: dict) -> str:
    rules = []

    rules.append(
        """\
## GENERAL RULES
- Extract text EXACTLY as it appears. Do NOT fix spelling, grammar, or abbreviations.
- Output ONLY raw HTML: no markdown, no code fences, no comments, no explanations.
- Do NOT use <style> blocks or global CSS. Use inline styles only where needed.
- Do NOT embed images as base64. Replace any stamp/image with: <img src='' alt='[image]'/>
- Preserve ALL text — do not skip, summarise, or truncate.
"""
    )

    font = (
        "Times New Roman, serif"
        if doc.get("font_style") == "serif"
        else "Arial, sans-serif"
    )
    align = "justify" if doc.get("text_alignment") in ("justify", "mixed") else "left"
    rules.append(
        f"""\
## DOCUMENT WRAPPER
- Wrap the entire document in:
  <div style='font-family:{font}; font-size:12pt; line-height:1.5; margin:2cm 2.5cm; text-align:{align};'>
- Close with </div> at the very end.
"""
    )

    if doc.get("has_letterhead"):
        border = (
            "<div style='border-bottom:3px solid black; padding-bottom:8px; margin-bottom:12pt;'>"
            if doc.get("has_horizontal_line_below_header")
            else "<div style='margin-bottom:12pt;'>"
        )
        rules.append(
            f"""\
## LETTERHEAD (centered)
- {border}
  <div style='text-align:center;'>
    <strong style='font-size:14pt;'>INSTITUTION NAME</strong><br/>
    <strong>SUB TITLE</strong><br/>
    <small>Address</small>
  </div>
  </div>
"""
        )

    if doc.get("heading_levels", 0) > 0:
        max_h = min(doc["heading_levels"] + 1, 4)
        rules.append(
            f"""\
## HEADINGS  ({doc['heading_levels']} level(s))
- Map heading hierarchy to <h1>...<h{max_h}>.
- CRITICAL: If a heading is visually centered, you MUST include `style='text-align:center;'`.
- If a heading is BOTH centered AND underlined, combine them exactly like this:
  <h2 style='text-align:center; margin-bottom:2pt;'><u>TEXT</u></h2>
- Reference line directly below heading (Nomor / No.):
  <p style='text-align:center; margin-top:0; margin-bottom:6pt;'>Nomor : ...</p>
"""
        )

    indent = (
        "- Indented paragraph: <p style='text-indent:40px; margin:0; margin-bottom:6pt;'>"
        if doc.get("has_indented_paragraphs")
        else ""
    )
    rules.append(
        f"""\
## PARAGRAPHS
- Standard paragraph: <p style='margin:0; margin-bottom:6pt;'>
{indent}
- Preserve visible blank lines as <br/> (max 2 consecutive).
"""
    )

    fmt = []
    if doc.get("has_bold_text"):
        fmt.append("Bold → <strong>")
    if doc.get("has_italic_text"):
        fmt.append("Italic → <em>")
    if doc.get("has_underlined_text"):
        fmt.append("Underline → <u>")
    if fmt:
        rules.append(
            "## TEXT FORMATTING\n- "
            + "\n- ".join(fmt)
            + "\n- Combine when needed: <strong><u>TEXT</u></strong>\n"
        )

    if doc.get("has_key_value_block"):
        if doc.get("has_nested_key_value"):
            ex_label, ex_html = "Nested KV", KV_NESTED_EXAMPLE
        elif doc.get("key_value_multiline"):
            ex_label, ex_html = "Multi-line KV", KV_MULTILINE_EXAMPLE
        else:
            ex_label, ex_html = "Standard KV", KV_SIMPLE_EXAMPLE

        multiline = (
            """\
  MULTI-LINE VALUE RULE:
  - If a label (e.g. Menimbang) has sub-items (1. 2. 3.), put ALL sub-items
    inside the value <td>, separated by <br/>. Do NOT use <ol>/<li>.
  - The label row appears ONCE — do not repeat label for each sub-item.
"""
            if doc.get("key_value_multiline")
            else ""
        )

        nested = (
            """\
  NESTED KV RULE:
  - Personal data fields (Nama, Jenis Kelamin, Tempat Tanggal Lahir, Jabatan, etc.)
    appearing INDENTED under a main label value → create a SECOND borderless <table>
    INSIDE the value <td>.
"""
            if doc.get("has_nested_key_value")
            else ""
        )

        rules.append(
            f"""\
## KEY-VALUE / COLON-ALIGNED BLOCKS
- Use a borderless table for every colon-aligned field block.
- ALL related labels go inside ONE <table> (not one table per row).
- Label column: white-space:nowrap — labels must never wrap.
- NEVER use &nbsp; chains or tabs for alignment.
{multiline}{nested}
EXAMPLE ({ex_label}):
{ex_html}
"""
        )

    if doc.get("is_handwritten"):
        rules.append(
            f"""\
## HANDWRITING RECOGNITION & EXTRACTION
- This document contains HANDWRITTEN text ({doc.get('handwriting_style', 'mixed')} style).
- Be extremely precise in deciphering it. If the script is cursive, use surrounding context to resolve ambiguous characters.
- If a word is truly illegible, represent it with [?] or your best phonetic guess.
- Maintain the visual placement of handwritten text even if it doesn't align perfectly with a grid.
- If there are handwritten annotations or notes in the margins, include them in the relevant part of the document using <p> or <div>.
"""
        )

    if doc.get("has_hand_drawn_table"):
        rules.append(
            """\
## HAND-DRAWN TABLES
- Even if the table lines are wobbly, hand-drawn, or incomplete, extract the content into a proper HTML <table>.
- Infer the number of rows and columns based on the content grouping and visual grid.
"""
        )

    if doc.get("has_data_table"):
        merged = (
            """\
- CRITICAL TABLE RULES:
  1. Top-left columns (e.g., "No", "Nama", "Unit Kerja") MUST have `rowspan='2'`.
  2. Group headers MUST have `colspan='X'`.
  3. The SECOND <tr> in <thead> MUST NOT contain any `rowspan`.
  EXAMPLE:
  <thead>
    <tr>
      <th rowspan='2'>No</th>
      <th rowspan='2'>Name</th>
      <th colspan='3'>Main Category</th>
    </tr>
    <tr>
      <th>Sub 1</th>
      <th>Sub 2</th>
      <th>Sub 3</th>
    </tr>
  </thead>
"""
            if doc.get("table_has_merged_cells")
            else ""
        )

        rules.append(
            f"""\
## DATA TABLE  (bordered grid)
- <table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse; width:100%;'>
- <thead> for header rows, <tbody> for data rows.
{merged}
"""
        )

    list_items = []
    style_val = doc.get("list_style", "none")
    if doc.get("has_numbered_list") and style_val == "sequential":
        list_items.append(
            "Sequential list (1,2,3) → <ol style='margin:0; padding-left:20px;'> with <li>."
        )
    elif doc.get("has_numbered_list"):
        list_items.append(
            "Irregular list → <p style='margin:0; margin-bottom:4pt;'> with number in text."
        )
    if doc.get("has_bullet_list"):
        list_items.append(
            "Bullet list → <ul style='margin:0; padding-left:20px;'> with <li>."
        )
    if doc.get("has_dash_list"):
        list_items.append(
            "Dash list → <p style='margin:0; margin-bottom:4pt;'>- Item text</p>. No <ul>."
        )
    if list_items:
        rules.append("## LISTS\n- " + "\n- ".join(list_items) + "\n")

    if doc.get("has_footer"):
        rules.append(
            """\
## FOOTER
- <div style='border-top:1px solid black; margin-top:12pt; padding-top:4px; font-size:10pt;'>
  footer text </div>
"""
        )

    if doc.get("has_bottom_dual_column"):
        tembusan = (
            """\
- LEFT column: Tembusan block using <p> tags with plain numbers (no <ol>/<li>).
"""
            if doc.get("has_tembusan_block")
            else ""
        )
        rules.append(
            f"""\
## BOTTOM DUAL-COLUMN  (Tembusan left  +  Date + Signature right)
CRITICAL: Tembusan and signature MUST be side by side using display:flex.

EXACT STRUCTURE:
{DUAL_BOTTOM_EXAMPLE}

{tembusan}Replace placeholder text with ACTUAL content from the image.
"""
        )

    elif doc.get("has_signature_block"):
        sig = doc.get("signature_position", "right")
        if sig == "dual":
            rules.append(
                """\
## SIGNATURE BLOCKS  (two side by side)
- <div style='display:flex; justify-content:space-between; margin-top:12pt;'>
    <div style='text-align:center; min-width:180px;'>LEFT BLOCK</div>
    <div style='text-align:center; min-width:180px;'>RIGHT BLOCK</div>
  </div><div style='clear:both;'></div>
"""
            )
        elif sig == "center":
            rules.append(
                """\
## SIGNATURE BLOCK  (centred)
- <div style='text-align:center; margin-top:12pt;'>
    Title<br/><br/><br/><strong>BOLD NAME</strong><br/>NIP
  </div>
"""
            )
        elif sig == "left":
            # FITUR BARU: Tanda tangan di kiri
            rules.append(
                """\
## SIGNATURE BLOCK  (left)
- <div style='float:left; text-align:center; margin-top:12pt; min-width:200px;'>
    Title<br/><br/><br/><strong>BOLD NAME</strong><br/>NIP
  </div><div style='clear:both;'></div>
"""
            )
        else:
            rules.append(
                """\
## SIGNATURE BLOCK  (right)
- <div style='float:right; text-align:center; margin-top:12pt; min-width:200px;'>
    Title<br/><br/><br/><strong>BOLD NAME</strong><br/>NIP
  </div><div style='clear:both;'></div>
"""
            )
    elif doc.get("has_place_date_line"):
        rules.append(
            """\
## PLACE / DATE LINE
- <div style='float:right; margin-top:12pt; min-width:260px;'>
    <table border='0' cellpadding='0' cellspacing='2'>
      <tr><td style='white-space:nowrap; padding-right:6px;'>Ditetapkan di</td>
          <td>:</td><td style='padding-left:6px;'>CITY</td></tr>
      <tr><td style='white-space:nowrap; padding-right:6px;'>Tanggal</td>
          <td>:</td><td style='padding-left:6px;'>DATE</td></tr>
    </table>
  </div>
"""
        )

    rules.append(
        """\
## SPACING
- Preserve visible blank lines with <br/> (max 2 consecutive).
- Do NOT add blank lines not in the original. Do NOT collapse section spacing.
"""
    )

    doc_type = doc.get("document_type", "document").replace("_", " ").upper()
    complexity = doc.get("complexity", "moderate").upper()
    return (
        f"/no_think\n"
        f"You are an expert OCR and document layout specialist.\n"
        f"Convert the document image (type:{doc_type}, complexity:{complexity})\n"
        f"into complete structure-preserving HTML.\n\n"
        f"CRITICAL:\n"
        f"- Follow ALL rules and EXAMPLE snippets exactly.\n"
        f"- Output ONLY raw HTML: no explanations, no prose, no code fences.\n"
        f"- Do NOT omit any text. Preserve all visual positions and indentation.\n\n"
        + "\n".join(rules)
    )


# ══════════════════════════════════════════════
# PHASE 3 — HTML Extraction  (with retry)
# ══════════════════════════════════════════════


def _validate_html(html: str, doc: dict) -> list:
    issues = []
    if doc.get("has_key_value_block") and "<table border='0'" not in html:
        issues.append(
            "Missing borderless KV table — colon-aligned fields not in a table."
        )
    if doc.get("has_bottom_dual_column") and "display:flex" not in html:
        issues.append(
            "Missing flex container — Tembusan and signature must be side by side."
        )
    if doc.get("has_tembusan_block") and re.search(r"Tembusan.*?<ol", html, re.DOTALL):
        issues.append(
            "Tembusan items inside <ol> — must use <p> tags with plain numbers."
        )

    # FITUR BARU: AUTO-REJECT TABEL HEADER YANG SALAH KAMAR
    if doc.get("table_has_merged_cells"):
        head_match = re.search(
            r"<thead[^>]*>(.*?)</thead", html, re.DOTALL | re.IGNORECASE
        )
        if head_match:
            thead = head_match.group(1)
            trs = re.findall(r"<tr[^>]*>(.*?)</tr", thead, re.DOTALL | re.IGNORECASE)
            if len(trs) > 1:
                # Cek jika baris kedua (tanggal) disusupi rowspan
                if "rowspan" in trs[1].lower():
                    issues.append(
                        "CRITICAL ERROR: You placed 'rowspan' inside the SECOND <tr> of the <thead>. This is structurally invalid. Move the 'rowspan' to the standalone columns ('No', 'Nama Guru', 'Unit Kerja') in the FIRST <tr>."
                    )
                # Cek jika baris pertama lupa dikasih rowspan
                if "rowspan" not in trs[0].lower():
                    issues.append(
                        "CRITICAL ERROR: You forgot to put 'rowspan=\"2\"' on the main standalone columns ('No', 'Nama Guru', 'Unit Kerja') in the FIRST <tr> of the <thead>."
                    )
    return issues


def phase3_extract(prompt: str, image_b64: str, doc: dict) -> str:
    for attempt in range(1, MAX_RETRIES + 2):
        tag = f"[Phase 3] Attempt {attempt}/{MAX_RETRIES + 1}"
        print(f"{tag} Extracting HTML ...")
        messages = [{"role": "user", "content": prompt, "images": [image_b64]}]
        raw = vision_chat(messages, max_tokens=MAX_TOKENS)
        html = strip_html_fences(raw)
        issues = _validate_html(html, doc)
        if not issues:
            print(f"{tag} OK")
            return html
        print(f"{tag} Validation issues:")
        for iss in issues:
            print(f"  - {iss}")
        if attempt <= MAX_RETRIES:
            feedback = (
                "The previous HTML output had these structural problems:\n"
                + "\n".join(f"  * {i}" for i in issues)
                + "\n\nFix ALL issues and regenerate complete HTML. "
                "Follow the rules and EXAMPLE snippets exactly. Output ONLY raw HTML."
            )
            prompt = prompt + "\n\n" + feedback
    print("[Phase 3] Max retries reached — returning best available output.")
    return html


# ══════════════════════════════════════════════
# HTML page wrapper
# ══════════════════════════════════════════════


def wrap_html_page(body_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Document OCR Result</title>
  <style>
    body  {{ background:#e0e0e0; margin:0; padding:20px; font-size:12pt; }}
    .page {{ background:white; max-width:820px; margin:0 auto;
             box-shadow:0 2px 12px rgba(0,0,0,0.25); }}
  </style>
</head>
<body>
  <div class="page">
    {body_html}
  </div>
</body>
</html>"""


# ══════════════════════════════════════════════
# DOCX Conversion
# ══════════════════════════════════════════════

def convert_to_docx(html: str, docx_path: str) -> str:
    """Convert HTML string to .docx using html_to_docx module."""
    try:
        # 1. Coba import relatif (Saat dijalankan oleh FastAPI)
        from .html_to_docx import html_to_docx
    except ImportError:
        try:
            # 2. Fallback import biasa (Saat script dijalankan sendirian)
            from html_to_docx import html_to_docx
        except ImportError as e:
            raise RuntimeError(
                f"Import error: {e}\n"
                "html_to_docx.py not found or missing dependencies.\n"
                "Ensure: pip install python-docx beautifulsoup4 lxml"
            )
            
    print(f"\n[DOCX] Converting HTML → {docx_path} ...")
    out = html_to_docx(html, docx_path)
    print(f"[DOCX] OK  Saved → {out}")
    return out

# ══════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════


def run_pipeline(
    image_path: str,
    output_path: str = None,
    save_prompt: bool = False,
    wrap_page: bool = True,
    make_docx: bool = True,
) -> dict:
    """
    Run the full pipeline.
    Returns dict with keys: html_path, docx_path, analysis_path.
    """
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Document OCR Pipeline v3.1  |  {Path(image_path).name}")
    print(f"{sep}\n")

    print("[Init] Loading image ...")
    image_b64 = load_image_base64(image_path)
    print(f"[Init] OK  ({len(image_b64)//1024} KB base64)\n")

    # Phase 1
    doc = phase1_analyse(image_b64)
    print("[Phase 1] Raw analysis:")
    print(json.dumps(doc, indent=2, ensure_ascii=False))

    # Phase 1.5
    doc = phase15_validate(doc, image_b64)
    print("\n[Phase 1.5] Final analysis:")
    print(json.dumps(doc, indent=2, ensure_ascii=False))

    # -------------------------------------------

    # Phase 2
    print("\n[Phase 2] Building extraction prompt ...")
    prompt = build_extraction_prompt(doc)
    print(f"[Phase 2] OK  ({len(prompt):,} chars)")
    if save_prompt:
        p = Path(image_path).stem + "_prompt.txt"
        Path(p).write_text(prompt, encoding="utf-8")
        print(f"[Phase 2] Prompt → {p}")

    print("\n-- Prompt preview (first 500 chars) " + "-" * 24)
    print(prompt[:500] + " ...")
    print("-" * 62 + "\n")

    # Phase 3
    html_body = phase3_extract(prompt, image_b64, doc)
    final_html = wrap_html_page(html_body) if wrap_page else html_body

    # Determine output paths
    parent_dir = Path(image_path).parent # Ambil direktori temp_dir
    stem = Path(image_path).stem
    html_out = output_path or (stem + "_output.html")
    docx_out = str(Path(html_out).with_suffix(".docx"))
    analysis_out = str(parent_dir / f"{stem}_analysis.json")

    # Save HTML
    Path(html_out).write_text(final_html, encoding="utf-8")

    # Save analysis
    Path(analysis_out).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Convert to DOCX
    if make_docx:
        convert_to_docx(final_html, docx_out)

    print(f"\n{'─'*62}")
    print(f"  Results:")
    print(f"  HTML     → {html_out}  ({len(final_html):,} chars)")
    if make_docx:
        sz = Path(docx_out).stat().st_size // 1024
        print(f"  DOCX     → {docx_out}  ({sz} KB)")
    print(f"  Analysis → {analysis_out}")
    print(f"{'─'*62}\n")

    return {
        "html_path": html_out,
        "docx_path": docx_out if make_docx else None,
        "analysis_path": analysis_out,
    }



# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════


def main():
    global OLLAMA_URL, MODEL

    parser = argparse.ArgumentParser(
        description="Document OCR Pipeline v3.1 — Image → HTML → DOCX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python doc_ocr_pipeline.py --image surat.jpg\n"
            "  python doc_ocr_pipeline.py --api\n"
            "  python doc_ocr_pipeline.py --docx-only result.html\n"
        ),
    )
    parser.add_argument(
        "--image", "-i", default=None, help="Path to the document image (jpg/png/webp)"
    )
    parser.add_argument(
        "--api", action="store_true", help="Start the Flask API server"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port for the API server (default: 5000)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output HTML path (default: <stem>_output.html)",
    )
    parser.add_argument(
        "--save-prompt",
        action="store_true",
        help="Save the generated Phase-2 prompt to a .txt file",
    )
    parser.add_argument(
        "--no-wrap",
        action="store_true",
        help="Output raw HTML fragment (no <html>/<body> wrapper)",
    )
    parser.add_argument("--no-docx", action="store_true", help="Skip DOCX conversion")
    parser.add_argument(
        "--docx-only",
        default=None,
        metavar="HTML_FILE",
        help="Convert an existing HTML file to DOCX (skip OCR)",
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_URL,
        help=f"Ollama API URL (default: {OLLAMA_URL})",
    )
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Model (default: {OLLAMA_MODEL})")

    args = parser.parse_args()
    OLLAMA_URL = args.ollama_url
    MODEL = args.model



    # ── DOCX-only mode ────────────────────────────────────────────────────
    if args.docx_only:
        src = Path(args.docx_only)
        if not src.exists():
            print(f"[Error] File not found: {src}", file=sys.stderr)
            sys.exit(1)
        html = src.read_text(encoding="utf-8")
        docx_path = str(src.with_suffix(".docx"))
        try:
            convert_to_docx(html, docx_path)
        except RuntimeError as e:
            print(f"[Error] {e}", file=sys.stderr)
            sys.exit(1)
        return

    # ── Full pipeline mode ────────────────────────────────────────────────
    if not args.image:
        parser.error("--image is required unless using --docx-only")

    if not Path(args.image).exists():
        print(f"[Error] Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        run_pipeline(
            image_path=args.image,
            output_path=args.output,
            save_prompt=args.save_prompt,
            wrap_page=not args.no_wrap,
            make_docx=not args.no_docx,
        )
    except requests.ConnectionError:
        print(f"\n[Error] Cannot connect to Ollama at {OLLAMA_URL}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n[Error] {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Aborted]", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
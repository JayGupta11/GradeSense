"""
GradeSense - Production OCR Module
===================================

Qwen2.5-VL-3B-Instruct based handwritten answer-sheet OCR.

Designed for:
    - Streamlit
    - Multiple student PDFs
    - Different handwriting
    - Different numbers of attempted questions
    - Out-of-order answers
    - MCQ + subjective pages
    - 4 GB RTX 3050 Laptop GPU
    - 4-bit BitsAndBytes quantization

Important:
    OCR reports only what it detects.
    It does NOT assume that every question was attempted.
"""

import os
import re
import gc
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "Qwen2.5-VL-3B-Instruct"
)

MODEL_PATH = os.environ.get(
    "GRADESENSE_QWEN_MODEL_PATH",
    DEFAULT_MODEL_PATH
).strip()

if not MODEL_PATH:
    MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"


# ============================================================
# VISION SETTINGS
# ============================================================

# Larger than the previous 1024 setting so handwritten text
# receives more visual tokens.
#
# 4 GB GPU means we should not push this too high.

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


# ============================================================
# GENERATION SETTINGS
# ============================================================

MAX_NEW_TOKENS_CLASSIFY = 20

MAX_NEW_TOKENS_MCQ_FULL = 700

MAX_NEW_TOKENS_MCQ_REGION = 500

MAX_NEW_TOKENS_SUBJECTIVE_FULL = 1500

MAX_NEW_TOKENS_SUBJECTIVE_REGION = 1100


# ============================================================
# GPU SETTINGS
# ============================================================

GPU_MEMORY_LIMIT = "3.55GiB"
CPU_MEMORY_LIMIT = "11GiB"


# ============================================================
# DETECTION SETTINGS
# ============================================================

# The model first sees the entire page.
#
# Then it sees overlapping horizontal bands.
#
# This makes handwriting physically larger to the vision model
# without assuming how many questions the student attempted.

REGION_OVERLAP = 0.18

MCQ_BAND_HEIGHT = 0.32

SUBJECTIVE_BAND_HEIGHT = 0.38


# ============================================================
# GLOBAL ENGINE
# ============================================================

_engine = None


# ============================================================
# LOGGING
# ============================================================

def _log(message: str):
    print(
        f"[GradeSense OCR] {message}",
        flush=True
    )


def _clear_cuda_cache():
    gc.collect()

    if torch.cuda.is_available():

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _log_vram():

    if not torch.cuda.is_available():
        return

    try:

        allocated = (
            torch.cuda.memory_allocated(0)
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved(0)
            / (1024 ** 3)
        )

        peak = (
            torch.cuda.max_memory_allocated(0)
            / (1024 ** 3)
        )

        _log(
            f"VRAM allocated: {allocated:.2f} GB | "
            f"reserved: {reserved:.2f} GB | "
            f"peak: {peak:.2f} GB"
        )

    except Exception:
        pass


# ============================================================
# CACHED QWEN MODEL
# ============================================================

@st.cache_resource(
    show_spinner=False,
    max_entries=1
)
def _load_qwen_cached(model_path: str):

    global _engine

    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    _log("=" * 70)
    _log("Loading Qwen2.5-VL-3B-Instruct")
    _log("=" * 70)

    _log(
        f"Model source: {model_path}"
    )

    if torch.cuda.is_available():

        gpu_name = torch.cuda.get_device_name(0)

        _log(
            f"GPU: {gpu_name}"
        )

        _log(
            f"CUDA: {torch.version.cuda}"
        )

        _log(
            "Loading 4-bit BitsAndBytes model..."
        )

        _log(
            f"GPU memory budget: "
            f"{GPU_MEMORY_LIMIT}"
        )

        _log(
            f"CPU offload budget: "
            f"{CPU_MEMORY_LIMIT}"
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        max_memory = {
            0: GPU_MEMORY_LIMIT,
            "cpu": CPU_MEMORY_LIMIT,
        }

        try:

            model = (
                Qwen2_5_VLForConditionalGeneration
                .from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory=max_memory,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                )
            )

        except TypeError:

            model = (
                Qwen2_5_VLForConditionalGeneration
                .from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory=max_memory,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
            )

        _engine = (
            "qwen2.5-vl-3b-instruct-4bit-cuda"
        )

    else:

        _log(
            "CUDA unavailable. "
            "Using CPU."
        )

        model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(
                model_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        )

        _engine = (
            "qwen2.5-vl-3b-instruct-cpu"
        )

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        use_fast=False,
    )

    model.eval()

    if torch.cuda.is_available():

        try:

            allocated = (
                torch.cuda.memory_allocated(0)
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved(0)
                / (1024 ** 3)
            )

            _log(
                f"GPU memory allocated: "
                f"{allocated:.2f} GB"
            )

            _log(
                f"GPU memory reserved: "
                f"{reserved:.2f} GB"
            )

        except Exception:
            pass

    _log(
        "Qwen model loaded and cached."
    )

    return model, processor


def _load_qwen():

    return _load_qwen_cached(
        MODEL_PATH
    )


# ============================================================
# IMAGE MESSAGE
# ============================================================

def _prepare_messages(
    image_path: str,
    prompt: str
):

    absolute_path = os.path.abspath(
        image_path
    )

    file_uri = (
        "file://"
        + absolute_path.replace(
            "\\",
            "/"
        )
    )

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": file_uri,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


# ============================================================
# QWEN INFERENCE
# ============================================================

def _generate_from_image(
    image_path: str,
    prompt: str,
    max_new_tokens: int
) -> str:

    model, processor = _load_qwen()

    from qwen_vl_utils import (
        process_vision_info
    )

    messages = _prepare_messages(
        image_path,
        prompt
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = (
        process_vision_info(
            messages
        )
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():

        inputs = inputs.to(
            "cuda"
        )

    else:

        inputs = inputs.to(
            "cpu"
        )

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    generated_ids_trimmed = [
        output_ids[
            len(input_ids):
        ]
        for input_ids, output_ids
        in zip(
            inputs.input_ids,
            generated_ids
        )
    ]

    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs
    del generated_ids
    del generated_ids_trimmed

    _clear_cuda_cache()

    return result.strip()


# ============================================================
# PAGE CLASSIFICATION
# ============================================================

CLASSIFICATION_PROMPT = r"""
Classify this student's answer-sheet page.

Return exactly one word:

MCQ

or

SUBJECTIVE

Choose MCQ when the page mainly contains multiple-choice
answers such as handwritten A, B, C, D or option text.

Choose SUBJECTIVE when the page mainly contains written
explanations, definitions, calculations, descriptive answers,
tables, diagrams, or long-form responses.

Do not explain.

Return only MCQ or SUBJECTIVE.
"""


def _classify_page(
    image_path: str
) -> str:

    raw = _generate_from_image(
        image_path,
        CLASSIFICATION_PROMPT,
        MAX_NEW_TOKENS_CLASSIFY
    )

    cleaned = raw.upper().strip()

    if re.search(
        r"\bMCQ\b",
        cleaned
    ):

        page_type = "MCQ"

    elif re.search(
        r"\bSUBJECTIVE\b",
        cleaned
    ):

        page_type = "SUBJECTIVE"

    else:

        page_type = "SUBJECTIVE"

    _log(
        f"Detected page type: {page_type}"
    )

    _log(
        f"Classification raw output: {raw!r}"
    )

    return page_type


# ============================================================
# OUTPUT CLEANING
# ============================================================

def _clean_raw_output(
    text: str
) -> str:

    if not text:
        return ""

    text = (
        text
        .replace(
            "\r\n",
            "\n"
        )
        .replace(
            "\r",
            "\n"
        )
        .strip()
    )

    text = re.sub(
        r"^\s*```(?:text|markdown|json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text
    )

    forbidden_lines = [
        r"literal\s+transcription",
        r"actual\s+handwritten\s+text",
        r"student's\s+actual\s+handwritten\s+answer",
        r"transcription",
        r"answer\s+key",
    ]

    for pattern in forbidden_lines:

        text = re.sub(
            rf"(?im)^\s*{pattern}\s*:?\s*$",
            "",
            text
        )

    return text.strip()


# ============================================================
# NUMBERED ANSWER PARSER
# ============================================================

def _parse_numbered_blocks(
    text: str
) -> List[Tuple[str, str]]:

    if not text:
        return []

    lines = (
        text
        .replace(
            "\r\n",
            "\n"
        )
        .replace(
            "\r",
            "\n"
        )
        .split("\n")
    )

    blocks = []

    current_number = None
    current_lines = []

    number_pattern = re.compile(
        r"^\s*"
        r"(?:Q(?:uestion)?\s*)?"
        r"(\d{1,3})"
        r"\s*"
        r"(?:[\.\):\-]|(?=\s))"
        r"\s*"
        r"(.*)$",
        re.IGNORECASE
    )

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        if line.upper() in (
            "[NO_ANSWERS]",
            "[NO ANSWERS]",
            "[NO ANSWER]",
            "NO ANSWERS",
            "NO ANSWERS DETECTED",
            "[NOT_FOUND]",
        ):
            continue

        match = number_pattern.match(
            line
        )

        if match:

            if current_number is not None:

                body = " ".join(
                    current_lines
                ).strip()

                if body:

                    blocks.append(
                        (
                            current_number,
                            body
                        )
                    )

            current_number = (
                match.group(1)
            )

            current_lines = []

            remainder = (
                match.group(2)
                .strip()
            )

            if remainder:

                current_lines.append(
                    remainder
                )

        else:

            if current_number is not None:

                current_lines.append(
                    line
                )

    if current_number is not None:

        body = " ".join(
            current_lines
        ).strip()

        if body:

            blocks.append(
                (
                    current_number,
                    body
                )
            )

    return blocks


# ============================================================
# QUESTION NUMBER VALIDATION
# ============================================================

def _valid_question_number(
    number: str
) -> bool:

    if not number:
        return False

    try:

        value = int(
            number
        )

    except Exception:

        return False

    return (
        1 <= value <= 200
    )


# ============================================================
# MCQ ANSWER NORMALIZATION
# ============================================================

def _clean_answer(
    answer: str
) -> str:

    answer = answer.strip()

    answer = answer.strip(
        "*_` "
    )

    answer = re.sub(
        r"\s+",
        " ",
        answer
    )

    return answer.strip()


def _normalize_mcq_answer(
    body: str
) -> str:

    body = _clean_answer(
        body
    )

    if not body:
        return ""

    # Examples:
    #
    # A
    # A.
    # A) Random Forest
    # A Random Forest
    #
    match = re.match(
        r"^([A-Da-d])"
        r"(?:[\.\):,\-]|\s|$)"
        r"(.*)$",
        body
    )

    if match:

        letter = (
            match.group(1)
            .upper()
        )

        remainder = (
            match.group(2)
            .strip()
        )

        if not remainder:

            return letter

        return (
            f"{letter}. "
            f"{remainder}"
        )

    # Handles forms such as:
    #
    # 3. C
    # C Random Forest Regression
    #
    return body


# ============================================================
# MCQ EXTRACTION
# ============================================================

def _extract_mcq_answers(
    text: str
) -> Dict[str, str]:

    answers = {}

    blocks = _parse_numbered_blocks(
        text
    )

    for number, body in blocks:

        number = str(
            number
        )

        if not _valid_question_number(
            number
        ):
            continue

        answer = _normalize_mcq_answer(
            body
        )

        if not answer:
            continue

        answers[
            number
        ] = answer

    return answers


# ============================================================
# MCQ PROMPT - WHOLE PAGE
# ============================================================

MCQ_FULL_PROMPT = r"""
Read this student's answer-sheet page carefully.

Your job is HANDWRITTEN ANSWER DETECTION.

Do NOT solve any question.

Do NOT use an answer key.

Do NOT guess.

Do NOT infer skipped questions.

Do NOT assume questions are answered sequentially.

The student may have answered only one MCQ, several MCQs,
or all MCQs.

Report ONLY handwritten answers that are actually visible.

For each answer, identify the question number written beside
the student's answer.

Supported examples include:

1. C
1. A
5. b
7. Random Forest Regression
12. 3. C
18. C Random Forest Regression

If the student wrote an option letter, preserve it.

If the student wrote option text, preserve the visible text.

If both are visible, preserve both.

Do NOT copy printed question options.

Do NOT copy printed answer choices.

Do NOT treat printed A/B/C/D choices as student answers.

Do NOT invent missing question numbers.

Do NOT fill gaps.

Do NOT create answers for blank questions.

Answers can be out of order.

Return ONLY detected handwritten answers, one per line.

Example:

2. B
7. D
14. A

The example is only formatting guidance. Do not copy it.

If there are no clearly detectable handwritten MCQ answers:

[NO_ANSWERS]
"""


# ============================================================
# MCQ PROMPT - REGION
# ============================================================

MCQ_REGION_PROMPT = r"""
This image is a crop from a student's MCQ answer sheet.

Detect handwritten MCQ answers visible in THIS CROP.

This is a recovery OCR pass.

IMPORTANT:

Only report a question if its question number is visibly
written in this crop.

Do NOT invent question numbers.

Do NOT continue a question number from another part of the page.

Do NOT assume the previous or next number.

Do NOT turn printed option numbers into student answers.

Do NOT turn printed A/B/C/D choices into student answers.

Do NOT turn random handwriting into an answer unless a
question number is visibly associated with it.

The student can answer questions in any order.

The student can leave any number of MCQs blank.

Return only answers with a clearly visible question number.

Examples:

4. A
11. C
18. Random Forest Regression

These examples are formatting examples only.

If the crop contains only continuation text and no clearly
visible question number belonging to a handwritten answer,
return:

[NO_ANSWERS]

Do not solve anything.

Do not grade anything.

Do not correct handwriting.

Do not explain.

Return only numbered answers.
"""


# ============================================================
# SUBJECTIVE FULL PROMPT
# ============================================================

SUBJECTIVE_FULL_PROMPT = r"""
Transcribe the student's handwritten answers on this page.

This is OCR only.

Do NOT grade.

Do NOT solve.

Do NOT correct spelling.

Do NOT correct grammar.

Do NOT improve wording.

Do NOT infer missing words.

Do NOT invent question numbers.

The student may answer questions out of order.

The student may skip any number of questions.

Only report a question number when that number is actually
visible beside the handwritten answer.

If an answer continues across many lines, keep the continuation
under the SAME question number.

Do NOT split one answer into multiple questions.

Ignore printed question text.

Ignore printed answer choices.

Ignore printed page numbers.

Ignore teacher markings.

Transcribe student handwriting only.

Preserve the student's wording and spelling.

Preserve bullet points when possible.

Preserve handwritten tables as clearly as possible.

Do not write explanations about the OCR.

Do not write the phrase "literal transcription".

Return ONLY numbered handwritten answers.

If there are no clearly visible numbered handwritten answers:

[NO_ANSWERS]
"""


# ============================================================
# SUBJECTIVE REGION PROMPT
# ============================================================

SUBJECTIVE_REGION_PROMPT = r"""
This image is a crop from a student's handwritten answer sheet.

Perform recovery OCR.

ONLY report an answer when the question number itself is
actually visible in this crop.

This rule is extremely important.

If the crop begins in the middle of an existing answer, do NOT
invent a question number for that continuation.

For example, if the crop shows:

- HTML, XML, JSON
- Need to do pre-processing
- Format in relational database

but the question number is NOT visible,

do NOT output:

21. ...
22. ...

Instead, return [NO_ANSWERS] unless another clearly numbered
answer is visible.

If the crop clearly shows:

31. k is numerical of cluster...

then report Q31.

If the crop clearly shows:

27. Data pre-processing...
28. We need normalization...

report both.

Students may answer questions out of order.

Students may skip questions.

Do NOT solve.

Do NOT grade.

Do NOT correct grammar.

Do NOT correct spelling.

Do NOT copy printed questions.

Do NOT copy printed options.

Do NOT turn table rows into questions.

Do NOT turn bullet points into questions.

Do NOT create sequential question numbers.

Do NOT continue numbering from outside the crop.

Return ONLY clearly numbered handwritten answers.

If no clearly numbered handwritten answer is visible:

[NO_ANSWERS]
"""


# ============================================================
# TEMPORARY CROP
# ============================================================

def _make_crop(
    image_path: str,
    y1: int,
    y2: int,
    label: str
) -> str:

    from PIL import Image

    image = Image.open(
        image_path
    )

    width, height = image.size

    y1 = max(
        0,
        min(
            height - 1,
            int(y1)
        )
    )

    y2 = max(
        y1 + 1,
        min(
            height,
            int(y2)
        )
    )

    crop = image.crop(
        (
            0,
            y1,
            width,
            y2
        )
    )

    temp_dir = tempfile.gettempdir()

    crop_path = os.path.join(
        temp_dir,
        (
            "gradesense_"
            + label
            + "_"
            + str(os.getpid())
            + ".png"
        )
    )

    crop.save(
        crop_path,
        format="PNG"
    )

    return crop_path


# ============================================================
# ADAPTIVE REGION GENERATION
# ============================================================

def _calculate_regions(
    height: int,
    band_height_ratio: float,
    overlap_ratio: float
) -> List[Tuple[int, int]]:

    if height <= 0:
        return []

    band_height = max(
        1,
        int(
            height
            * band_height_ratio
        )
    )

    overlap = int(
        band_height
        * overlap_ratio
    )

    step = max(
        1,
        band_height - overlap
    )

    regions = []

    start = 0

    while start < height:

        end = min(
            height,
            start + band_height
        )

        regions.append(
            (
                start,
                end
            )
        )

        if end >= height:
            break

        start += step

    # Remove duplicate regions.

    unique = []

    seen = set()

    for region in regions:

        if region in seen:
            continue

        seen.add(
            region
        )

        unique.append(
            region
        )

    return unique


# ============================================================
# MCQ FULL PAGE
# ============================================================

def _mcq_full_page(
    image_path: str
):

    raw = _generate_from_image(
        image_path,
        MCQ_FULL_PROMPT,
        MAX_NEW_TOKENS_MCQ_FULL
    )

    raw = _clean_raw_output(
        raw
    )

    answers = _extract_mcq_answers(
        raw
    )

    _log(
        f"Whole-page MCQ answers: "
        f"{len(answers)}"
    )

    _log(
        "Whole-page MCQ output:\n"
        + (
            raw
            if raw
            else "[NO_ANSWERS]"
        )
    )

    return answers


# ============================================================
# MCQ REGION OCR
# ============================================================

def _mcq_regions(
    image_path: str
):

    from PIL import Image

    image = Image.open(
        image_path
    )

    width, height = image.size

    regions = _calculate_regions(
        height,
        MCQ_BAND_HEIGHT,
        REGION_OVERLAP
    )

    _log(
        f"Created {len(regions)} "
        f"overlapping MCQ regions."
    )

    observations = []

    for index, (
        y1,
        y2
    ) in enumerate(
        regions,
        start=1
    ):

        _log(
            f"MCQ region {index}: "
            f"Y={y1}->{y2}"
        )

        crop_path = _make_crop(
            image_path,
            y1,
            y2,
            f"mcq_{index}"
        )

        try:

            raw = _generate_from_image(
                crop_path,
                MCQ_REGION_PROMPT,
                MAX_NEW_TOKENS_MCQ_REGION
            )

            raw = _clean_raw_output(
                raw
            )

            answers = _extract_mcq_answers(
                raw
            )

            _log(
                f"MCQ region {index} answers: "
                f"{len(answers)}"
            )

            _log(
                f"MCQ region {index} output:\n"
                + (
                    raw
                    if raw
                    else "[NO_ANSWERS]"
                )
            )

            observations.append(
                answers
            )

        finally:

            try:

                if os.path.exists(
                    crop_path
                ):

                    os.remove(
                        crop_path
                    )

            except Exception:
                pass

    return observations


# ============================================================
# MCQ MERGE
# ============================================================

def _merge_mcq(
    full_answers: Dict[str, str],
    region_answers: List[Dict[str, str]]
):

    observations = defaultdict(
        list
    )

    # Whole page gets highest trust.

    for number, answer in (
        full_answers.items()
    ):

        observations[
            str(number)
        ].append(
            (
                answer,
                100
            )
        )

    # Region observations.

    for region in region_answers:

        for number, answer in (
            region.items()
        ):

            observations[
                str(number)
            ].append(
                (
                    answer,
                    50
                )
            )

    merged = {}

    for number, candidates in (
        observations.items()
    ):

        if not candidates:
            continue

        # Prefer the candidate seen by the
        # whole-page pass.

        full_candidates = [
            item
            for item in candidates
            if item[1] == 100
        ]

        if full_candidates:

            answer = max(
                full_candidates,
                key=lambda x: len(
                    x[0]
                )
            )[0]

        else:

            # Count agreement among regions.

            counts = defaultdict(
                int
            )

            for answer, _ in candidates:

                counts[
                    answer.lower().strip()
                ] += 1

            best_key = max(
                counts,
                key=counts.get
            )

            matching = [
                answer
                for answer, _ in candidates
                if answer.lower().strip()
                == best_key
            ]

            answer = max(
                matching,
                key=len
            )

        merged[
            number
        ] = answer

    return merged


# ============================================================
# MCQ MASTER
# ============================================================

def _process_mcq(
    image_path: str
) -> str:

    from PIL import Image

    image = Image.open(
        image_path
    )

    width, height = image.size

    _log(
        f"MCQ page size: "
        f"{width} x {height}"
    )

    # --------------------------------------------------------
    # Pass 1: whole page
    # --------------------------------------------------------

    full_answers = _mcq_full_page(
        image_path
    )

    # --------------------------------------------------------
    # Pass 2:
    #
    # Always use region OCR for detection quality.
    #
    # This is intentionally NOT based on a fixed number of
    # expected questions.
    # --------------------------------------------------------

    region_answers = _mcq_regions(
        image_path
    )

    merged = _merge_mcq(
        full_answers,
        region_answers
    )

    _log(
        f"Final merged MCQ answers: "
        f"{len(merged)}"
    )

    return _format_answers(
        merged
    )


# ============================================================
# SUBJECTIVE FULL PAGE
# ============================================================

def _subjective_full_page(
    image_path: str
):

    raw = _generate_from_image(
        image_path,
        SUBJECTIVE_FULL_PROMPT,
        MAX_NEW_TOKENS_SUBJECTIVE_FULL
    )

    raw = _clean_raw_output(
        raw
    )

    blocks = _parse_numbered_blocks(
        raw
    )

    _log(
        f"Whole-page subjective answers: "
        f"{len(blocks)}"
    )

    _log(
        "Whole-page subjective output:\n"
        + (
            raw
            if raw
            else "[NO_ANSWERS]"
        )
    )

    return blocks


# ============================================================
# SUBJECTIVE REGION OCR
# ============================================================

def _subjective_regions(
    image_path: str
):

    from PIL import Image

    image = Image.open(
        image_path
    )

    width, height = image.size

    regions = _calculate_regions(
        height,
        SUBJECTIVE_BAND_HEIGHT,
        REGION_OVERLAP
    )

    _log(
        f"Created {len(regions)} "
        f"overlapping subjective regions."
    )

    results = []

    for index, (
        y1,
        y2
    ) in enumerate(
        regions,
        start=1
    ):

        _log(
            f"Subjective region {index}: "
            f"Y={y1}->{y2}"
        )

        crop_path = _make_crop(
            image_path,
            y1,
            y2,
            f"subjective_{index}"
        )

        try:

            raw = _generate_from_image(
                crop_path,
                SUBJECTIVE_REGION_PROMPT,
                MAX_NEW_TOKENS_SUBJECTIVE_REGION
            )

            raw = _clean_raw_output(
                raw
            )

            blocks = _parse_numbered_blocks(
                raw
            )

            _log(
                f"Subjective region {index} "
                f"answers: {len(blocks)}"
            )

            _log(
                f"Subjective region {index} output:\n"
                + (
                    raw
                    if raw
                    else "[NO_ANSWERS]"
                )
            )

            results.append(
                blocks
            )

        finally:

            try:

                if os.path.exists(
                    crop_path
                ):

                    os.remove(
                        crop_path
                    )

            except Exception:
                pass

    return results


# ============================================================
# SUBJECTIVE BODY CLEANING
# ============================================================

def _clean_subjective_body(
    body: str
) -> str:

    if not body:
        return ""

    body = body.strip()

    body = re.sub(
        r"(?im)^\s*literal\s+transcription\s*$",
        "",
        body
    )

    body = re.sub(
        r"(?im)^\s*actual\s+handwritten\s+text\s*$",
        "",
        body
    )

    body = re.sub(
        r"\n{3,}",
        "\n\n",
        body
    )

    return body.strip()


# ============================================================
# FRAGMENT DETECTION
# ============================================================

def _looks_like_fragment(
    body: str
) -> bool:

    if not body:
        return True

    text = body.strip()

    if len(text) < 3:
        return True

    lower = text.lower()

    # Typical pieces that can occur when a crop begins
    # inside a previous answer.

    fragment_only = {
        "html, xml, json",
        "html xml json",
        "agriculture",
        "hospitalities",
        "analyst.",
        "analyst",
        "rational database",
        "relational database",
    }

    if lower in fragment_only:
        return True

    return False


# ============================================================
# SUBJECTIVE MERGE
# ============================================================

def _merge_subjective(
    whole_blocks,
    region_results
):

    candidates = defaultdict(
        list
    )

    # --------------------------------------------------------
    # Whole-page observations
    # --------------------------------------------------------

    for number, body in whole_blocks:

        number = str(
            number
        )

        body = _clean_subjective_body(
            body
        )

        if not _valid_question_number(
            number
        ):
            continue

        if not body:
            continue

        if _looks_like_fragment(
            body
        ):
            continue

        candidates[
            number
        ].append(
            (
                body,
                100
            )
        )

    # --------------------------------------------------------
    # Region observations
    # --------------------------------------------------------

    for blocks in region_results:

        for number, body in blocks:

            number = str(
                number
            )

            body = _clean_subjective_body(
                body
            )

            if not _valid_question_number(
                number
            ):
                continue

            if not body:
                continue

            if _looks_like_fragment(
                body
            ):
                _log(
                    f"Rejected likely fragment "
                    f"Q{number}: {body}"
                )

                continue

            candidates[
                number
            ].append(
                (
                    body,
                    50
                )
            )

    # --------------------------------------------------------
    # Final selection
    # --------------------------------------------------------

    merged = {}

    source_info = {}

    for number, entries in (
        candidates.items()
    ):

        if not entries:
            continue

        full_entries = [
            entry
            for entry in entries
            if entry[1] == 100
        ]

        if full_entries:

            # Whole-page answer is preferred,
            # but select the longest complete
            # whole-page transcription.

            best = max(
                full_entries,
                key=lambda x: len(
                    x[0]
                )
            )

            merged[
                number
            ] = best[0]

        else:

            # Region-only question.
            #
            # If several regions saw it, choose
            # the longest observation.

            best = max(
                entries,
                key=lambda x: len(
                    x[0]
                )
            )

            merged[
                number
            ] = best[0]

        source_info[
            number
        ] = {
            "whole_page": bool(
                full_entries
            ),
            "observations": len(
                entries
            ),
        }

    return merged, source_info


# ============================================================
# SUBJECTIVE MASTER
# ============================================================

def _process_subjective(
    image_path: str
) -> str:

    whole_blocks = (
        _subjective_full_page(
            image_path
        )
    )

    region_results = (
        _subjective_regions(
            image_path
        )
    )

    merged, source_info = (
        _merge_subjective(
            whole_blocks,
            region_results
        )
    )

    numbers = sorted(
        merged.keys(),
        key=lambda x: (
            0,
            int(x)
        )
        if x.isdigit()
        else (
            1,
            x
        )
    )

    _log(
        "Subjective merged question numbers: "
        + (
            ", ".join(
                numbers
            )
            if numbers
            else "[NONE]"
        )
    )

    for number in numbers:

        info = source_info.get(
            number,
            {}
        )

        _log(
            f"Q{number}: "
            f"whole_page="
            f"{info.get('whole_page', False)}, "
            f"observations="
            f"{info.get('observations', 0)}"
        )

    return _format_subjective_answers(
        merged
    )


# ============================================================
# FORMATTING
# ============================================================

def _format_answers(
    answers: Dict[str, str]
) -> str:

    def sort_key(item):

        number = item[0]

        if str(number).isdigit():

            return (
                0,
                int(number)
            )

        return (
            1,
            str(number)
        )

    output = []

    for number, answer in sorted(
        answers.items(),
        key=sort_key
    ):

        answer = _clean_answer(
            answer
        )

        if answer:

            output.append(
                f"{number}. {answer}"
            )

    return "\n".join(
        output
    )


def _format_subjective_answers(
    answers: Dict[str, str]
) -> str:

    def sort_key(number):

        if str(number).isdigit():

            return (
                0,
                int(number)
            )

        return (
            1,
            str(number)
        )

    output = []

    for number in sorted(
        answers.keys(),
        key=sort_key
    ):

        body = (
            answers[number]
            .strip()
        )

        if body:

            output.append(
                f"{number}. {body}"
            )

    return "\n\n".join(
        output
    )


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def extract_text(
    image_path: str,
    preprocess: bool = True
) -> str:
    """
    Main OCR entry point used by GradeSense.
    """

    if not image_path:
        return ""

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"OCR image not found: "
            f"{image_path}"
        )

    try:

        if torch.cuda.is_available():

            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        page_type = _classify_page(
            image_path
        )

        if page_type == "MCQ":

            result = _process_mcq(
                image_path
            )

        else:

            result = _process_subjective(
                image_path
            )

        _log_vram()

        return result.strip()

    except torch.cuda.OutOfMemoryError as exc:

        _clear_cuda_cache()

        raise RuntimeError(
            "Qwen OCR ran out of GPU memory. "
            "Reduce MAX_PIXELS from "
            "1280*28*28 to 1024*28*28 "
            "in ocr_module.py."
        ) from exc


# ============================================================
# COMPATIBILITY API
# ============================================================

def extract_text_column_aware(
    image_path: str,
    preprocess: bool = True
) -> str:

    return extract_text(
        image_path,
        preprocess=preprocess
    )


def extract_text_with_boxes(
    image_path: str,
    preprocess: bool = True
):

    text = extract_text(
        image_path,
        preprocess=preprocess
    )

    return text, []


# ============================================================
# PRINTED QUESTION PAPER / ANSWER KEY OCR
# ============================================================

PRINTED_TEXT_PROMPT = r"""
Read all printed or typed text visible in this image.

This may be a question paper or answer key.

Transcribe the visible content accurately.

Preserve:

- question numbers
- section headings
- marks
- option letters
- option text
- answer-key entries
- tables
- equations

Do not solve anything.

Do not add explanations.

Return only the transcription.
"""


def extract_printed_text(
    image_path: str,
    preprocess: bool = True
) -> str:

    if not image_path:
        return ""

    if not os.path.exists(
        image_path
    ):

        raise FileNotFoundError(
            f"Printed OCR image not found: "
            f"{image_path}"
        )

    result = _generate_from_image(
        image_path,
        PRINTED_TEXT_PROMPT,
        MAX_NEW_TOKENS_SUBJECTIVE_FULL
    )

    return result.strip()


# ============================================================
# ENGINE INFORMATION
# ============================================================

def get_active_engine():

    return _engine


def get_model_info():

    gpu = None

    if torch.cuda.is_available():

        try:

            gpu = torch.cuda.get_device_name(
                0
            )

        except Exception:
            gpu = None

    return {
        "model": MODEL_PATH,
        "engine": _engine,
        "cuda": torch.cuda.is_available(),
        "gpu": gpu,
        "4bit": torch.cuda.is_available(),
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "mcq_band_height":
            MCQ_BAND_HEIGHT,
        "subjective_band_height":
            SUBJECTIVE_BAND_HEIGHT,
        "region_overlap":
            REGION_OVERLAP,
        "model_loaded":
            _engine is not None,
    }


# ============================================================
# EXPLICIT MODEL UNLOAD
# ============================================================

def unload_model():

    global _engine

    try:

        _load_qwen_cached.clear()

    except Exception:
        pass

    _engine = None

    _clear_cuda_cache()

    _log(
        "Qwen model cache cleared."
    )


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            'python ocr_module.py '
            '"C:\\path\\to\\image.jpg"'
        )

        raise SystemExit(1)

    image_path = sys.argv[1]

    print()
    print(
        "=" * 70
    )

    print(
        "GradeSense Qwen OCR Test"
    )

    print(
        "=" * 70
    )

    result = extract_text(
        image_path
    )

    print()
    print(
        "OCR RESULT"
    )

    print(
        "=" * 70
    )

    print(
        result
        if result
        else "[NO ANSWERS DETECTED]"
    )

    print(
        "=" * 70
    )

    print(
        f"Engine: "
        f"{get_active_engine()}"
    )

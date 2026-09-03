# GradeSense

AI-assisted handwritten answer sheet evaluation system using Qwen2.5-VL-3B-Instruct, Streamlit, and MongoDB.

## Quick Setup

Run these steps **in order**.

### 1. Install requirements

From the project root:

```powershell
python -m venv haesenv
.\haesenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` installs the project dependencies and the **PyTorch CUDA 12.6 build**.

Verify:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
```

Expected:

```text
CUDA available: True
CUDA runtime: 12.6
```

An NVIDIA GPU and compatible NVIDIA driver are required. A separate CUDA Toolkit installation is normally **not required** to run GradeSense because the PyTorch CUDA wheel provides its CUDA runtime.

If CPU-only/incompatible PyTorch is already installed:

```powershell
pip uninstall torch torchvision torchaudio -y
```

Then run:

```powershell
pip install -r requirements.txt
```

### 2. Set up Qwen dependencies

```powershell
cd src
python setup_qwen.py
```

### 3. Download Qwen model

```powershell
python download_qwen_model.py
```

The model will be stored at:

```text
models/Qwen2.5-VL-3B-Instruct
```

After these three steps, the environment and Qwen model are ready.

### 4. Gmail OTP Configuration

GradeSense uses Gmail SMTP for OTP verification.

Open the file containing the SMTP/OTP configuration and update:

EMAIL_ADDRESS = "your-email@gmail.com"
EMAIL_APP_PASSWORD = "your-16-character-app-password"

#### Create a Gmail App Password

1. Open your Google Account.

2. Go to Security.

3. Enable 2-Step Verification.

4. Open App Passwords.

5. Create a new password for GradeSense.

6. Copy the generated 16-character App Password.

7. Use it in the application instead of your normal Gmail password.

Run GradeSense:

```powershell
streamlit run app_batch.py
```

Open:

```text
http://localhost:8501
```

---

## Project Structure

```text
GradeSense/
├── data/
├── models/
│   └── Qwen2.5-VL-3B-Instruct/
├── src/
│   ├── assets/
│   │    └── logo.png
│   ├── rubic_scoring.py
│   ├── app_batch.py
│   ├── auth.py
│   ├── database.py
│   ├── ocr_module.py
│   ├── batch_grade.py
│   ├── pdf_utils.py
│   ├── setup_qwen.py
│   ├── download_qwen_model.py
│   ├── test_environment.py
│   ├── pipeline.py
│   ├── train_marks_predictor.py
│   ├── scoring_engine.py
│   ├── paper_parser.py
│   ├── nlp_evaluator.py
│   ├── marks_predictor.py
│   ├── email_utils.py
│   ├── choice_grouping.py
│   └── blur_detection.py
├── requirements.txt
└── README.md
```

---

## MongoDB Integration

GradeSense uses **MongoDB** for:

- User accounts and authentication
- Email verification/OTP
- Evaluation history
- Student evaluation results

Make sure MongoDB is installed and running.

Configure the MongoDB connection using the project's database configuration/environment variables.

1. Check MongoDB Installation

Open PowerShell and run:

mongod --version

If MongoDB is installed, its version will be displayed.

Check whether the MongoDB service is available:

Get-Service MongoDB

2. Start MongoDB

If the MongoDB service shows Stopped, start it:

net start MongoDB

3. Check MongoDB Shell

Test whether MongoDB Shell (mongosh) is installed:

mongosh --version

If the version is displayed, open MongoDB Shell:

mongosh

You should see a connection similar to:

Connecting to: mongodb://127.0.0.1:27017/

then type exit

---

## Environment Test

After setup:

```powershell
cd src
python test_environment.py
```

This checks Python, PyTorch, CUDA, GPU, Transformers, Accelerate, BitsAndBytes, and qwen-vl-utils.

## Qwen Model

Model:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Download it with:

```powershell
python download_qwen_model.py
```

Do not commit the downloaded model to GitHub because of its size.

## Tested Environment

```text
Python 3.13.7
PyTorch CUDA 12.6
NVIDIA RTX 3050 Laptop GPU
16 GB RAM
Transformers 5.16.1
Accelerate 1.14.0
BitsAndBytes 0.50.2
```

Run GradeSense:

```powershell
streamlit run app_batch.py
```
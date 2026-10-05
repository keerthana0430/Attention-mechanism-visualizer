# AI Attention Visualizer

A beginner-friendly AI application that extracts text from study-note images using Optical Character Recognition (OCR), converts words into numerical embeddings, applies a basic Attention Mechanism, and visually displays the attention scores of words.

The project is designed as an educational application to help students understand how OCR, text embeddings, and the Attention Mechanism can work together in a single AI application.

---

## Project Overview

The **AI Attention Visualizer** takes a study-notes image as input and processes it through multiple AI steps.

The application:

1. Accepts an image containing study notes.
2. Extracts text from the image using Tesseract OCR.
3. Splits the extracted text into individual words.
4. Converts the words into numerical embeddings using Sentence Transformers.
5. Applies a basic scaled dot-product Attention Mechanism.
6. Calculates relative attention scores for each word.
7. Displays the scores using Streamlit progress bars.
8. Identifies the word with the highest calculated attention score.

### Project Flow

```text
Image
   ↓
OCR
   ↓
Extracted Text
   ↓
Word Processing
   ↓
Sentence Embeddings
   ↓
Query, Key, Value
   ↓
Scaled Dot-Product Attention
   ↓
Attention Scores
   ↓
Streamlit Visualization
```

---

## Project Objective

The main objective of this project is to build a simple AI application that demonstrates the basic working of the Attention Mechanism.

The project combines multiple concepts:

* Optical Character Recognition
* Text Processing
* Sentence Embeddings
* Query, Key and Value
* Scaled Dot-Product Attention
* Attention Weights
* Streamlit Visualization

The visualization makes the concept easier to understand without requiring users to study complicated attention matrices.

---

## Problem Statement

Students often learn OCR, embeddings, and attention as separate concepts.

This project combines these concepts into one simple application.

A user uploads an image containing study notes. The application extracts the text, processes the words, generates embeddings, calculates attention scores, and displays the attention values visually.

---

## Technologies Used

| Technology            | Purpose                              |
| --------------------- | ------------------------------------ |
| Python                | Main programming language            |
| Streamlit             | Web application interface            |
| Tesseract OCR         | Extract text from images             |
| pytesseract           | Python interface for Tesseract       |
| Sentence Transformers | Generate text embeddings             |
| NumPy                 | Numerical and attention calculations |
| Pillow                | Open and process images              |
| Visual Studio Code    | Development environment              |

---

## Features

* Upload JPG, JPEG, and PNG images
* Extract text from study-note images
* Display extracted text
* Clean and process extracted words
* Generate numerical word embeddings
* Calculate attention scores
* Display word-level attention bars
* Identify the highest-attended word
* Simple Streamlit interface
* Beginner-friendly AI demonstration

---

## Development Environment

The project was developed using:

```text
Visual Studio Code
Python 3.x
Windows
```

VS Code is used to create the Python files, install packages through the terminal, run the Streamlit application, and test the project.

---

## Project Structure

```text
AI-Attention-Visualizer/
│
├── app.py
├── ocr.py
├── embedding.py
├── attention.py
├── requirements.txt
└── README.md
```

### File Description

**app.py**

Main Streamlit application. It handles image uploading, displays extracted text, generates embeddings, calculates attention scores, and displays the visualization.

**ocr.py**

Contains the OCR functionality using Tesseract and pytesseract.

**embedding.py**

Uses the `all-MiniLM-L6-v2` Sentence Transformer model to generate numerical representations of words.

**attention.py**

Implements the basic scaled dot-product attention calculation using NumPy.

**requirements.txt**

Contains the Python libraries required to run the project.

**README.md**

Contains project documentation and setup instructions.

---

# Installation and Setup

## 1. Install Python

Install Python 3.x on your Windows computer.

Check whether Python is installed:

```bash
python --version
```

---

## 2. Open the Project in VS Code

Open **Visual Studio Code**.

Open the project folder:

```text
AI-Attention-Visualizer
```

The folder should contain:

```text
app.py
ocr.py
embedding.py
attention.py
requirements.txt
README.md
```

---

## 3. Open VS Code Terminal

In VS Code:

```text
Terminal → New Terminal
```

---

## 4. Install Required Packages

Run:

```bash
pip install -r requirements.txt
```

Or install the packages individually:

```bash
pip install streamlit
pip install numpy
pip install pillow
pip install pytesseract
pip install sentence-transformers
```

---

# Tesseract OCR Setup

Tesseract OCR is required separately because `pytesseract` is only a Python wrapper.

Install Tesseract OCR on Windows.

A common installation location is:

```text
C:\Program Files\Tesseract-OCR\
```

The executable should be:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

The path is configured in `ocr.py`.

Example:

```python
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def extract_text(image):
    return pytesseract.image_to_string(image)
```

If Tesseract is installed in a different location, update the path accordingly.

---

# requirements.txt

The project requires:

```text
streamlit
numpy
pillow
pytesseract
sentence-transformers
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# Module 1: OCR

The OCR module extracts text from the uploaded image.

Example:

```python
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def extract_text(image):
    return pytesseract.image_to_string(image)
```

### How OCR Works

The user uploads an image.

Tesseract analyzes the image and identifies characters and words.

The extracted text is returned to the Python application.

```text
Image → Tesseract OCR → Text
```

---

# Module 2: Word Embeddings

The project uses the pretrained:

```text
all-MiniLM-L6-v2
```

Sentence Transformer model.

Example:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(words):
    embeddings = model.encode(words)
    return embeddings
```

An embedding is a numerical representation of text.

Instead of processing a word only as characters, the application represents it as a vector of numbers.

The project uses these numerical representations as input for the attention calculation.

---

# Module 3: Attention Mechanism

The Attention Mechanism is implemented using NumPy.

Basic implementation:

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(
        x - np.max(x, axis=-1, keepdims=True)
    )

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )
```

The attention calculation creates Query, Key and Value representations.

```python
def calculate_attention(X):

    d_model = X.shape[1]
    d_k = 64

    np.random.seed(42)

    WQ = np.random.randn(d_model, d_k)
    WK = np.random.randn(d_model, d_k)
    WV = np.random.randn(d_model, d_k)

    Q = X @ WQ
    K = X @ WK
    V = X @ WV

    scores = Q @ K.T

    scaled_scores = scores / np.sqrt(d_k)

    attention_weights = softmax(scaled_scores)

    word_scores = attention_weights.mean(axis=0)

    return word_scores
```

---

# Attention Formula

The basic attention formula used in the project is:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V
```

### Explanation

**Q - Query**

Represents what the current word is looking for.

**K - Key**

Represents information used to compare relationships between words.

**V - Value**

Contains the information that is passed through the attention mechanism.

**QKᵀ**

Calculates relationships between the representations.

**√dₖ**

Scales the attention scores.

**Softmax**

Converts the scores into attention weights.

The application averages the attention received by each word to obtain one display score for each word.

---

# Module 4: Streamlit Application

The Streamlit application connects all the modules.

Basic structure:

```python
import streamlit as st
import numpy as np
from PIL import Image

from ocr import extract_text
from embedding import create_embeddings
from attention import calculate_attention

st.title("AI Attention Visualizer")

file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if file:

    image = Image.open(file)

    st.image(image)

    text = extract_text(image)

    st.subheader("Extracted Text")

    st.write(text)
```

After extracting the text, the application processes the words and calculates attention.

```python
words = text.split()

embeddings = create_embeddings(words)

scores = calculate_attention(embeddings)
```

The scores are then normalized and displayed using Streamlit progress bars.

---

# How the Application Works

### Step 1: Upload Image

The user uploads a study-notes image.

Supported formats:

```text
JPG
JPEG
PNG
```

### Step 2: OCR

Tesseract OCR extracts text from the uploaded image.

```text
Image → Extracted Text
```

### Step 3: Word Processing

The extracted text is split into individual words.

Words are cleaned by removing punctuation.

Words shorter than three characters are removed.

The application uses a maximum of 20 words for the demonstration.

### Step 4: Embeddings

Each selected word is converted into a numerical representation using the Sentence Transformer model.

### Step 5: Attention

The embeddings are passed to the attention module.

Query, Key and Value representations are generated.

Scaled dot-product attention is then calculated.

### Step 6: Visualization

The attention scores are normalized and displayed using progress bars.

### Step 7: Highest Attention

The application identifies the word with the highest calculated attention score.

---

# Running the Application

Open the project in VS Code.

Open:

```text
Terminal → New Terminal
```

Run:

```bash
streamlit run app.py
```

The terminal will show a local URL.

Usually:

```text
http://localhost:8501
```

Open this URL in your browser.

---

# Example

Suppose the uploaded image contains:

```text
Artificial Intelligence uses machine learning to analyze data.
```

The application extracts the text:

```text
Artificial Intelligence uses machine learning to analyze data.
```

The words are converted into embeddings and passed through the attention mechanism.

The application then displays attention bars such as:

```text
Artificial       ██████████
Intelligence     ███████████████
uses             █████
machine          █████████
learning         █████████████
analyze          ███████
data             █████████
```

The word with the highest calculated score is displayed as the highest-attention word.

---

# Expected Output

The application displays:

1. Uploaded study-notes image
2. Extracted text
3. Word Attention section
4. Attention progress bars
5. Highest-attended word

Example:

```text
Uploaded Image
      ↓
Extracted Text
      ↓
Word Attention
      ↓
Artificial       ███████
Intelligence     ███████████
machine          █████████
learning         ████████████

Highest Attention: learning
```

The exact values depend on the input text and attention calculation.

---

# Testing

The project can be tested using different types of images.

### Test 1

Upload a clear image containing a short paragraph.

Expected result:

```text
Extracted text should be displayed.
```

### Test 2

Upload an image containing a heading and body text.

Expected result:

```text
Words should be extracted and attention bars should appear.
```

### Test 3

Upload an image containing handwritten text.

Expected result:

```text
OCR accuracy may vary.
```

### Test 4

Upload an image with no readable text.

Expected result:

```text
No text found in the image.
```

### Test 5

Upload an image containing more than 20 words.

Expected result:

```text
Only the first 20 cleaned words are visualized.
```

---

# Common Errors

### Streamlit not installed

```text
ModuleNotFoundError: No module named 'streamlit'
```

Solution:

```bash
pip install streamlit
```

### Sentence Transformers not installed

```text
ModuleNotFoundError: No module named 'sentence_transformers'
```

Solution:

```bash
pip install sentence-transformers
```

### Pytesseract not installed

```text
ModuleNotFoundError: No module named 'pytesseract'
```

Solution:

```bash
pip install pytesseract
```

### Tesseract not found

Check the path in `ocr.py`:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

### No text found

Use a clearer image with:

* Larger text
* Good lighting
* Better contrast
* Clear printed characters

---

# Project Limitations

* OCR accuracy depends on image quality.
* Handwritten text may not be recognized accurately.
* Only the first 20 cleaned words are visualized.
* The MiniLM model is used at word level for educational simplicity.
* Query, Key and Value projection matrices are randomly initialized.
* The highest-attention word should not be considered a reliable semantic importance detector.
* This project demonstrates the mechanics of attention rather than the actual internal attention maps of a pretrained Transformer.

---

# Future Enhancements

The project can be improved by adding:

* True Transformer token-level attention
* Multi-Head Attention visualization
* Query, Key and Value visualization
* Attention heatmaps
* Highlighting important words directly in the text
* OCR bounding-box visualization
* Tamil and other language support
* Keyword extraction
* Topic classification
* Study-notes summarization
* Question-answering from uploaded notes
* Downloadable attention reports

---

# Learning Outcomes

Through this project, students can understand:

* What OCR is
* How images can be converted into text
* What text embeddings are
* Why text is converted into numerical vectors
* The purpose of Query, Key and Value
* Scaled dot-product attention
* The role of Softmax
* Attention weights
* Streamlit application development
* How different AI components can be combined into one project

---

# Conclusion

The **AI Attention Visualizer** is a compact educational AI project that combines image processing, OCR, text embeddings, attention calculations, and a Streamlit web interface.

The project demonstrates the complete flow:

```text
Image
  ↓
OCR
  ↓
Text
  ↓
Embeddings
  ↓
Attention
  ↓
Visualization
```

It is designed to make the basic concept of the Attention Mechanism easier to understand through a simple and interactive application.

---

# Author

**Keerthana**

BSc Computer Science with Artificial Intelligence

Developed using **Python and Visual Studio Code**.

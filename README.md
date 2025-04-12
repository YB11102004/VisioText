# VisioText

**AI-powered tool for transforming text into stunning images, extracting and correcting text, and enhancing visuals seamlessly.**

## Features
- Generate images from text prompts using advanced AI models.
- Extract and correct text from generated or uploaded images.
- Overlay corrected text back onto images with professional aesthetics.

## Deployment
Access the application on [Hugging Face Spaces](https://huggingface.co/spaces/ybhimani/AI_Enhanced_Text_to_Image_with_Intelligent_Text_Correction).

## Installation
### Prerequisites
- Python 3.8+
- Tesseract OCR
- Required Python packages

### Steps
1. Clone the repository:
   ```bash
   git clone https://huggingface.co/spaces/ybhimani/AI_Enhanced_Text_to_Image_with_Intelligent_Text_Correction

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Add your API key for text correction: Replace your-api-key-here in the script:
   ```bash
   pip install -r requirements.txt
   client_gemini = genai.Client(api_key="your-api-key-here")

4. Run the application:
   ```bash
   streamlit run app.py

### Example
![Generated image vs Corrected image](https://drive.google.com/uc?id=1Ww8PWG0Vh3UqXRp8K2wpZKuOrebpgcsr)

### License
This project is licensed under the MIT License. See LICENSE for details.

### Acknowledgements
    black-forest-labs/FLUX.1-schnell for image generation.
    Google Gemini AI for text correction.
    Tesseract OCR for text extraction.
    
### Feedback and contributions are always welcome! 🌟

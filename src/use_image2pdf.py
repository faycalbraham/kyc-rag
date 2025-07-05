# src/use_image2pdf.py

from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf_images(pdf_path, poppler_path=r"C:\Program Files\poppler\Library\bin"):
    """
    Convertit un PDF scanné en images puis applique Tesseract OCR pour extraire le texte.
    Retourne la liste de textes par page.
    """
    texts = []
    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang="fra")
            texts.append(text)
            print(f"✅ OCR page {i+1} terminée ({len(text.strip())} caractères)")
    except Exception as e:
        print(f"❌ Erreur dans extract_text_from_pdf_images: {e}")
    return texts
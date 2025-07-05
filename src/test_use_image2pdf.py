# test_use_image2pdf.py

from use_image2pdf import extract_text_from_pdf_images

if __name__ == "__main__":
    pdf_path = "./data/Vos-Justificatifs-BNP-Paribas_QES.pdf"
    texts = extract_text_from_pdf_images(pdf_path)

    for i, page in enumerate(texts):
        print(f"--- Page {i+1} ---")
        print(page)  # affiche  toute la page 
        print()
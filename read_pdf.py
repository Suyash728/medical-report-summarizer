import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import io

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF. Uses OCR for scanned PDFs if normal extraction yields minimal text.
    """
    extracted_data = []
    
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        print(f'The PDF has {num_pages} pages.\n')

        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            
            # If text extraction yields little or no text, use OCR
            if not text or len(text.strip()) < 50:
                print(f'Page {page_num + 1}: Minimal text detected - running OCR...')
                try:
                    # Convert PDF page to image for OCR
                    images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
                    ocr_text = pytesseract.image_to_string(images[0])
                    text = ocr_text
                    print(f'  ✓ OCR completed')
                except Exception as e:
                    print(f'  ✗ OCR failed: {e}')
                    text = text or "[Unable to extract text]"
            
            extracted_data.append({
                'page': page_num + 1,
                'text': text
            })
            print(f'Page {page_num + 1}:\n{text}\n')

    return extracted_data

# Main execution
if __name__ == '__main__':
    pdf_file = 'sample2.pdf'
    try:
        data = extract_text_from_pdf(pdf_file)
        print("\n=== Extraction Complete ===")
    except FileNotFoundError:
        print(f"Error: {pdf_file} not found in current directory")
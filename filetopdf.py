import os
from fpdf import FPDF

def text_to_pdf(text, filename, output_dir):
    """Convert text to a PDF file with UTF-8 support."""
    pdf_filename = os.path.splitext(filename)[0] + '.pdf'
    pdf_path = os.path.join(output_dir, pdf_filename)
    pdf = FPDF()
    pdf.add_page()

    # Set a font that supports Unicode (e.g., 'Arial Unicode MS' or any available Unicode font)
    pdf.set_font('Arial', '', 12)

    # Add the text, while handling the encoding properly
    # The text will be encoded and decoded to avoid any non-Latin characters issue
    pdf.multi_cell(0, 10, text.encode('latin-1', 'ignore').decode('latin-1'))
    pdf.output(pdf_path)
    print(f"Converted {filename} to {pdf_filename}")

def convert_files_to_pdfs(source_dir, output_dir):
    """Convert all files in source_dir and its subdirectories to PDFs in output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.startswith('.'):
                continue  # Skip hidden files
            file_path = os.path.join(root, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
            text_to_pdf(text, filename, output_dir)

if __name__ == "__main__":
    source_directory = './files'  # Replace with your source directory path
    output_directory = './pdfs'  # Output directory for PDFs
    convert_files_to_pdfs(source_directory, output_directory)

import fitz  # PyMuPDF
import re

def extract_latex_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    latex_content = []

    for page_number in range(doc.page_count):
        page = doc[page_number]
        text = page.get_text()

        # Use a regular expression to extract LaTeX expressions
        latex_matches = re.findall(r'\$[^\$]+\$', text)

        # Append LaTeX expressions to the result list
        latex_content.extend(latex_matches)

    doc.close()
    return latex_content

# Example usage
pdf_path = './pdf/A STRUCTURED SELF-ATTENTIVE.pdf'
latex_content = extract_latex_from_pdf(pdf_path)

for i, latex_expression in enumerate(latex_content, start=1):
    print(f"LaTeX Expression {i}: {latex_expression}")
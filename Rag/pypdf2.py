import PyPDF2


def read_pdf(file_path):
    with open(file_path, "rb") as file:

        # Create a PDF Reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Get the number of pages in the PDF
        num_pages = len(pdf_reader.pages)

        for page_number in range(num_pages):
            page = pdf_reader.pages[page_number]

            text = page.extract_text()

            # Print the text from each page
            print(f'\nPage {page_number + 1}:\n{text}')


if __name__ == '__main__':
    pdf_file_path = './pdf/A STRUCTURED SELF-ATTENTIVE.pdf'

    read_pdf(pdf_file_path)
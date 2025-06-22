
EXTRACT_STRUCTURED_DATA_PROMPT = """
You are an expert data extraction AI. Your task is to analyze the following text from a PDF document and extract all relevant key-value pairs.
The document is a form, so pay attention to labels and the corresponding values filled in.
Present the extracted data as a clean JSON object. The keys of the JSON should be descriptive, snake_cased labels for the data, and the values should be the extracted information.
Clean up the keys to be descriptive and consistent (e.g., use "company_name" instead of "Name of the company").

Here is the text from the PDF:
---
{pdf_text}
---

Please return only the JSON object.
"""

GENERATE_SUMMARY_PROMPT = """Based on the following data extracted from the document, please generate a concise summary that captures the key information.
The summary should include important details such as company names, dates, and the nature of the filings or resolutions. It should be clear and easy to understand for a non-technical audience.
generate a 3-5 line summary for a non-technical person.
Data:
{json_data}
"""

ATTACHMENT_SUMMARY_PROMPT = """
You are an AI assistant. Please summarize the following text extracted from the attachments of a corporate filing.
The attachments likely include board resolutions, consent letters from auditors, etc.
Focus on key information like dates, names, and the nature of the resolutions or consents.

Attachment Text:
---
{attachment_text}
---

Provide a concise summary.
"""

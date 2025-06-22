import fitz
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path, max_pages=None):
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        num_pages = doc.page_count
        if max_pages:
            num_pages = min(num_pages, max_pages)

        for i in range(num_pages):
            page = doc[i]
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def extract_attachment_text(pdf_path):
    """Extracts text from pages considered to be attachments."""
    attachment_text = ""
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count > 1:
            for page_num in range(1, doc.page_count):
                attachment_text += doc[page_num].get_text()
        doc.close()
        return attachment_text
    except Exception as e:
        print(f"Error extracting attachment text: {e}")
        return ""

def extract_structured_data_with_llm(pdf_text):
    """Uses an LLM to extract structured data from raw text."""
    prompt = f"""
    You are an expert data extraction AI. Your task is to analyze the following text from a PDF document and extract all relevant key-value pairs.
    The document is a government form, so pay attention to labels and the corresponding values filled in.
    Present the extracted data as a clean JSON object. The keys of the JSON should be descriptive, snake_cased labels for the data, and the values should be the extracted information.
    Clean up the keys to be descriptive and consistent (e.g., use "company_name" instead of "Name of the company").

    Here is the text from the PDF:
    ---
    {pdf_text}
    ---

    Please return only the JSON object.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert data extraction AI that returns JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        response_text = response.choices[0].message.content.strip()
        
        # The model might return the JSON inside a code block, so we need to extract it.
        match = re.search(r'```json\n({.*?})\n```', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = response_text

        return json.loads(json_str)
    except Exception as e:
        print(f"Error during LLM data extraction: {e}")
        return None

def generate_summary(json_data, attachment_summary=None):
    """Generates a summary using OpenAI's GPT model."""
    prompt = f"""Based on the following data from a company's auditor appointment form (ADT-1),
generate a 3-5 line summary for a non-technical person.
Data:
{json.dumps(json_data, indent=2)}"""

    if attachment_summary:
        prompt += f"""

Also, consider the following summary from the attachments:
---
{attachment_summary}
---
Incorporate any important details from the attachments, such as board resolutions or consent letters, into the main summary.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes corporate filings."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

def summarize_attachments(attachment_text):
    """Uses an LLM to summarize the text of attachments."""
    prompt = f"""
    You are an AI assistant. Please summarize the following text extracted from the attachments of a corporate filing.
    The attachments likely include board resolutions, consent letters from auditors, etc.
    Focus on key information like dates, names, and the nature of the resolutions or consents.

    Attachment Text:
    ---
    {attachment_text}
    ---

    Provide a concise summary.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant that summarizes legal and corporate documents."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating attachment summary: {e}"

def main():
    pdf_path = "Form ADT-1-29092023_signed.pdf"
    
    # 1. Extract raw text from the main form
    form_text = extract_text_from_pdf(pdf_path)
    if not form_text:
        return

    # 2. Use LLM to extract structured data from the form
    structured_data = extract_structured_data_with_llm(form_text)
    if not structured_data:
        print("Could not extract structured data using LLM.")
        return

    # 3. Save JSON to file
    with open("output.json", "w") as f:
        json.dump(structured_data, f, indent=4)
    
    print("Extracted JSON data:")
    print(json.dumps(structured_data, indent=4))
    
    # 4. Process attachments if available
    attachment_summary = None
    attachment_text = extract_attachment_text(pdf_path)
    if attachment_text:
        print("\n--- Processing Attachments ---")
        attachment_summary = summarize_attachments(attachment_text)
        with open("attachment_summary.txt", "w") as f:
            f.write(attachment_summary)
        print("Attachment summary saved to attachment_summary.txt")
        print(f"\nAttachment Summary:\n{attachment_summary}")
    
    # Printing the list of attachments from the JSON
    attachments = structured_data.get("attachments")
    if attachments:
        print("\n--- Attachments Listed in Form ---")
        for attachment in attachments:
            print(f"- {attachment}")
    else:
        print("\nNo attachments listed in the form.")

    # 5. Generate AI summary
    summary = generate_summary(structured_data, attachment_summary)
    print("\nAI-generated summary:")
    print(summary)
    with open("summary.txt", "w") as f:
            f.write(summary)

if __name__ == "__main__":
    main()
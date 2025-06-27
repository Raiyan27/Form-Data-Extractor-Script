import fitz
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import prompts
from config import GPT_MODEL 

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
    prompt = prompts.EXTRACT_STRUCTURED_DATA_PROMPT.format(pdf_text=pdf_text)
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
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
    prompt = prompts.GENERATE_SUMMARY_PROMPT.format(json_data=json.dumps(json_data, indent=2))

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
            model=GPT_MODEL,
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
    prompt = prompts.ATTACHMENT_SUMMARY_PROMPT.format(attachment_text=attachment_text)
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an AI assistant that summarizes legal and corporate documents."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating attachment summary: {e}"

def main():
    input_dir = "files_to_extract"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in the '{input_dir}' directory.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        base_filename = os.path.splitext(pdf_file)[0]
        
        print(f"--- Processing {pdf_file} ---")

        # 1. Extract raw text from the main form
        form_text = extract_text_from_pdf(pdf_path)
        if not form_text:
            continue

        # 2. Use LLM to extract structured data from the form
        structured_data = extract_structured_data_with_llm(form_text)
        if not structured_data:
            print(f"Could not extract structured data for {pdf_file}.")
            continue

        # 3. Save JSON to file
        output_json_path = f"{base_filename}_output.json"
        with open(output_json_path, "w") as f:
            json.dump(structured_data, f, indent=4)
        
        print(f"Extracted JSON data saved to {output_json_path}")
        
        # 4. Process attachments if available
        attachment_summary = None
        attachment_text = extract_attachment_text(pdf_path)
        if attachment_text:
            print("\n--- Processing Attachments ---")
            attachment_summary = summarize_attachments(attachment_text)
            attachment_summary_path = f"{base_filename}_attachment_summary.txt"
            with open(attachment_summary_path, "w") as f:
                f.write(attachment_summary)
            print(f"Attachment summary saved to {attachment_summary_path}")
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
        summary_path = f"{base_filename}_summary.txt"
        print("\nAI-generated summary:")
        print(summary)
        with open(summary_path, "w") as f:
                f.write(summary)
        print(f"Summary saved to {summary_path}")
        print(f"--- Finished processing {pdf_file} ---\n")

if __name__ == "__main__":
    main()
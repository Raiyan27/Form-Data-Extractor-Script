import fitz 
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

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
            model="gpt-4-turbo",
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

def generate_summary(json_data):
    """Generates a summary using OpenAI's GPT model."""
    prompt = f"Based on the following data, generate a 3-5 line summary for a non-technical person:\n\n{json.dumps(json_data, indent=2)}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

def main():
    pdf_path = "Form ADT-1-29092023_signed.pdf"
    
    # 1. Extract raw text from PDF
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        return

    # 2. Use LLM to extract structured data
    structured_data = extract_structured_data_with_llm(raw_text)
    if not structured_data:
        print("Could not extract structured data using LLM.")
        return

    # 3. Save JSON to file
    with open("output.json", "w") as f:
        json.dump(structured_data, f, indent=4)
    
    print("Extracted JSON data:")
    print(json.dumps(structured_data, indent=4))
    
    # 4. Generate AI summary
    summary = generate_summary(structured_data)
    print("\nAI-generated summary:")
    print(summary)

    # 5. Extract attachments if possible
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        attachments = re.findall(r'Attach\n(.+\.pdf)', text)
        if attachments:
            print("\n--- Attachments ---")
            for attachment in attachments:
                print(f"- {attachment.strip()}")
        else:
            print("\nNo attachments found.")
    except Exception as e:
        print(f"Error processing attachments: {e}")

if __name__ == "__main__":
    main()
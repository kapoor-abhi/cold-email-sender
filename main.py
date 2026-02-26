import os
import base64
from typing import Optional, List
from enum import Enum

# FastAPI & Pydantic
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field, SecretStr

# LangChain & Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

# Gmail API
from email.message import EmailMessage
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Initialize FastAPI
app = FastAPI(title="Cold Email Agent")
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- CONFIGURATION ---
# Replace this with your actual Groq API Key or set it in your environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# --- PYDANTIC MODELS (The Brain) ---

class EmailGenerationOutput(BaseModel):
    """
    Strict model to ensure the LLM provides the email components separately.
    """
    recipient_email: str = Field(
        description="The email address of the hiring manager or company extracted from the job description. If none is found, return an empty string."
    )
    subject_line: str = Field(
        description="A professional, catchy subject line optimized for the job description or according to requested in the job desciption like send email using subject this 'Application Data Analyst'."
    )
    email_body: str = Field(
        description="The main content of the cold email. It must use 'Dear Hiring Manager' as the greeting."
    )

class UserRequest(BaseModel):
    """
    Model for the initial request from the user (if sending JSON instead of Form data).
    """
    user_email: str
    company_info: str
    role_info: str

print("Step 1: Imports and Models loaded successfully.")



llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b", 
    temperature=0.6 
)


parser = PydanticOutputParser(pydantic_object=EmailGenerationOutput)


email_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert career coach and professional copywriter specializing in highly effective cold emails."),
    ("human", """Write me a professional cold email optimized specifically for the company and aligned with their goals and the role description.

Company Information: {company_info}
Role/Job Description: {role_info}
My Resume/Profile: {resume_text}

The email must strictly follow these rules:
- ALWAYS start with 'Dear Hiring Manager,' as the greeting. DO NOT use any specific person's name.
- Start with a catchy, professional opening line that grabs attention.
- Be well-structured, concise, and engaging throughout (not too long or boring).
- Highlight my skills and relevant projects from my resume ONLY if they match the company’s needs.
- Maintain a professional yet interesting tone that keeps the reader’s interest.
- End with a clear, polite call-to-action (e.g., scheduling a call, collaboration, or follow-up).
- Ensure the email looks organized from top to bottom, with smooth flow and no unnecessary fluff.
- Extract the recipient email address from the Company Information or Role/Job Description. If none is found, leave it empty.
- The subject of the email should be tailored to what is demanded in the job description.

{format_instructions}
""")
])


email_generation_chain = email_prompt | llm | parser

print("Step 2: LangChain and Groq LLM setup complete.")


import io
from pypdf import PdfReader



async def extract_text_from_file(file: UploadFile) -> str:
    """Reads uploaded files and extracts text from PDFs or raw TXT."""
    if not file:
        return ""
    
    content = await file.read()
    
    if file.filename.lower().endswith(".pdf"):
        # Parse PDF in memory
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    else:
        # Assume it's a standard text file
        return content.decode('utf-8', errors='ignore')

# --- FASTAPI ENDPOINTS ---

@app.post("/generate_email", response_model=EmailGenerationOutput)
async def generate_email(
    resume: UploadFile = File(..., description="Upload resume as PDF or TXT"),
    job_description_file: Optional[UploadFile] = File(None, description="Optional: Upload Job Description as PDF/TXT"),
    job_description_text: Optional[str] = Form("", description="Optional: Paste Job Description text"),
    company_info: str = Form(..., description="Company goals, description, or target info")
):
    """
    Phase 1 of HITL: Generate the draft email and return it to the user for review.
    """

    resume_text = await extract_text_from_file(resume)
    
 
    jd_text = job_description_text
    if job_description_file:
        jd_text += "\n" + await extract_text_from_file(job_description_file)
        
    if not resume_text or not jd_text.strip():
        raise HTTPException(status_code=400, detail="Both Resume and Job Description are required.")


    try:
        draft_result = email_generation_chain.invoke({
            "company_info": company_info,
            "role_info": jd_text,
            "resume_text": resume_text,
            "format_instructions": parser.get_format_instructions()
        })
        return draft_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email generation failed: {str(e)}")

print("Step 3: Text extraction and /generate_email endpoint complete.")



from langchain_core.output_parsers import StrOutputParser


class RevisionRequest(BaseModel):
    current_draft: str
    feedback: str

# We use a simpler string parser here because we only need the rewritten body text back
revision_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert copywriter refining a cold email. You MUST always start the email with 'Dear Hiring Manager,'."),
    ("human", """Here is the current draft of the email:
{current_draft}

The user has provided the following feedback for changes:
{feedback}

Please rewrite the email incorporating this feedback. 
Maintain a professional tone, keep it concise, and ensure it flows well.
Do not include any extra commentary, just return the revised email body.
""")
])

revision_chain = revision_prompt | llm | StrOutputParser()

@app.post("/revise_email")
async def revise_email(request: RevisionRequest):
    """
    Phase 2 of HITL: Refine the email based on user feedback.
    """
    try:
        revised_body = revision_chain.invoke({
            "current_draft": request.current_draft,
            "feedback": request.feedback
        })
        return {"revised_body": revised_body}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Revision failed: {str(e)}")


# --- GMAIL API SENDING LOGIC ---

def get_gmail_service():
    """Authenticates and returns the Gmail service based on your working logic."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

@app.post("/send_email")
async def send_email(
    target_email: str = Form(..., description="The auto-detected or manually verified email"),
    subject: str = Form(..., description="The finalized subject line"),
    body: str = Form(..., description="The finalized email body"),
    resume: UploadFile = File(..., description="The resume to attach")
):
    """
    Final Phase: Send the email with the resume attached using Gmail API.
    """
    try:
        service = get_gmail_service()
        
        # 1. Construct the email message
        message = EmailMessage()
        message.set_content(body)
        message['To'] = target_email
        message['From'] = 'me' # 'me' refers to the authenticated user in Gmail API
        message['Subject'] = subject
        
        # 2. Handle the attachment
        resume_content = await resume.read()
        
        # Determine basic mime type for attachment
        maintype = 'application'
        subtype = 'pdf' if resume.filename.lower().endswith('.pdf') else 'octet-stream'
        if resume.filename.lower().endswith('.txt'):
            maintype, subtype = 'text', 'plain'
            
        message.add_attachment(
            resume_content,
            maintype=maintype,
            subtype=subtype,
            filename=resume.filename
        )

        
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {'raw': encoded_message}
        
        send_message = service.users().messages().send(userId="me", body=create_message).execute()
        
        return {"status": "Success", "message_id": send_message['id']}
        
    except HttpError as error:
        raise HTTPException(status_code=500, detail=f"Gmail API error: {error}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Cold Email Sender API is running!"}
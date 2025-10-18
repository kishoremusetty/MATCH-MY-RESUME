import os
import json 
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from google import genai
from google.genai import types
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
app = Flask(__name__)
# NOTE: To run this, you need to create a folder named 'templates' 
# and put the merged HTML file inside it, named 'index.html'.
app.config['UPLOAD_FOLDER'] = 'uploads' 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 1. Initialize Gemini Client
client = None
try:
    # Client will automatically pick up GEMINI_API_KEY from the .env file
    client = genai.Client()
    print("Gemini Client initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini Client: {e}")
    
# --- Helper Functions ---

def pdf_to_text(pdf_path):
    """Extracts all text from a local PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            # Handle potential None or non-string return from extract_text
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text

def generate_rewritten_resume(job_description, resume_text):
    """Calls the Gemini API to rewrite the resume."""
    if not client:
        return "Gemini API Error: Client not initialized."
        
    # Powerful prompt instruction for the model
    system_instruction = (
        "You are an expert HR and professional resume writer. "
        "Your task is to rewrite the provided 'Existing Resume' to perfectly match "
        "the 'Job Description'. Use existing content that is suitable. "
        "If any required skills or projects are missing for the job role, seamlessly add them based on common job requirements. "
        "Rephrase and reformat content to highlight keywords and relevance. "
        "Use bullet points and proper professional sections. "
        "Heading (like project names, technical skills, education, certification, and awards) should be in **bold and uppercase**."
        "The output must be the full, complete, and rewritten resume text, formatted clearly with Markdown. "
        "DO NOT include any introductory dialogue or surrounding text—only the resume. "
        "DO NOT use # symbols for headings - use **bold** formatting instead. "
        "DO NOT use --- separators anywhere in the resume. "
        "The resume can be longer than one page if necessary."
    )
    
    # Combine user inputs into the main prompt content
    user_content = (
        f"Job Description:\n---\n{job_description}\n---\n\n"
        f"Existing Resume:\n---\n{resume_text}\n---\n\n"
        "REWRITTEN PROFESSIONAL RESUME:"
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.6,
            ),
        )
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"

def generate_cover_letter(job_description, resume_text, template_style="professional"):
    """Calls the Gemini API to generate a cover letter."""
    if not client:
        return "Gemini API Error: Client not initialized."
    
    # Define different template styles
    template_instructions = {
        "professional": (
            "Write a professional, formal cover letter with clear structure including: "
            "proper business letter format with date, recipient address, greeting, "
            "3-4 paragraphs (introduction, body with specific examples, closing), "
            "and professional sign-off."
        ),
        "modern": (
            "Write a modern, engaging cover letter that stands out while remaining professional. "
            "Use a more conversational tone, include specific achievements with metrics, "
            "and demonstrate passion for the role. Structure: compelling opening, "
            "2-3 body paragraphs with concrete examples, and strong closing."
        ),
        "creative": (
            "Write a creative cover letter that showcases personality while maintaining professionalism. "
            "Use storytelling elements, include unique angles about why you're perfect for the role, "
            "and demonstrate creativity in presentation. Structure: hook opening, "
            "narrative body with examples, memorable closing."
        ),
        "executive": (
            "Write an executive-level cover letter with strategic focus and leadership emphasis. "
            "Highlight vision, strategic thinking, and high-level achievements. "
            "Use confident language and focus on business impact. Structure: "
            "executive summary opening, strategic body paragraphs, leadership-focused closing."
        )
    }
    
    template_prompt = template_instructions.get(template_style, template_instructions["professional"])
    
    # Powerful prompt instruction for the model
    system_instruction = (
        f"You are an expert professional cover letter writer. "
        f"{template_prompt} "
        "Extract relevant information from both the job description and resume to create a compelling cover letter. "
        "Use specific examples from the resume that match job requirements. "
        "Address the hiring manager directly and demonstrate knowledge of the company/role. "
        "The output must be the complete cover letter text, properly formatted. "
        "DO NOT include any introductory dialogue or surrounding text—only the cover letter. "
        "Use proper business letter formatting with appropriate spacing and structure."
    )
    
    # Combine user inputs into the main prompt content
    user_content = (
        f"Job Description:\n---\n{job_description}\n---\n\n"
        f"Resume Information:\n---\n{resume_text}\n---\n\n"
        "PROFESSIONAL COVER LETTER:"
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            ),
        )
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the home page."""
    return render_template('home.html')

@app.route('/ats-checker')
def ats_checker():
    """Renders the ATS Score Checker page."""
    return render_template('ats_checker.html')

@app.route('/resume-generator')
def resume_generator():
    """Renders the Resume Generator page."""
    return render_template('resume_generator.html')

@app.route('/cover-letter')
def cover_letter():
    """Renders the Cover Letter Generator page."""
    return render_template('cover_letter.html')

@app.route('/skill-gap')
def skill_gap():
    """Renders the Skill Gap Analysis page."""
    return render_template('skill_gap.html')

@app.route('/rewrite_resume', methods=['POST'])
def rewrite_resume():
    """Handles the form submission and Gemini API call."""
    if 'resume' not in request.files or 'job_description' not in request.form:
        return jsonify({'error': 'Missing file or job description'}), 400

    job_description = request.form['job_description']
    resume_file = request.files['resume']

    if resume_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if resume_file and resume_file.filename.endswith('.pdf'):
        # 1. Save and extract text from PDF
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(filepath)
        
        resume_text = pdf_to_text(filepath)
        
        if not resume_text:
            os.remove(filepath)
            return jsonify({'error': 'Could not extract text from PDF. The file might be corrupted or image-only.'}), 500

        # 2. Call Gemini
        rewritten_text = generate_rewritten_resume(job_description, resume_text)
        
        # 3. Clean up the uploaded file
        os.remove(filepath) 

        # 4. Return result
        if rewritten_text.startswith("Gemini API Error:"):
             return jsonify({'error': rewritten_text}), 500
             
        return jsonify({
            'rewritten_resume': rewritten_text,
            'original_resume': resume_text
        })

    return jsonify({'error': 'Unsupported file type. Please upload a PDF file.'}), 400

@app.route('/upload_resume_for_ats', methods=['POST'])
def upload_resume_for_ats():
    """Handles resume upload for ATS analysis only."""
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    resume_file = request.files['resume']
    if resume_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if resume_file and resume_file.filename.endswith('.pdf'):
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(filepath)
        
        resume_text = pdf_to_text(filepath)
        
        if not resume_text:
            os.remove(filepath)
            return jsonify({'error': 'Could not extract text from PDF. The file might be corrupted or image-only.'}), 500

        os.remove(filepath)
        return jsonify({'original_resume': resume_text})

    return jsonify({'error': 'Unsupported file type. Please upload a PDF file.'}), 400

@app.route('/upload_resume_for_cover_letter', methods=['POST'])
def upload_resume_for_cover_letter():
    """Handles resume upload for cover letter generation only."""
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    resume_file = request.files['resume']
    if resume_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if resume_file and resume_file.filename.endswith('.pdf'):
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(filepath)
        
        resume_text = pdf_to_text(filepath)
        
        if not resume_text:
            os.remove(filepath)
            return jsonify({'error': 'Could not extract text from PDF. The file might be corrupted or image-only.'}), 500

        os.remove(filepath)
        return jsonify({'resume_text': resume_text})

    return jsonify({'error': 'Unsupported file type. Please upload a PDF file.'}), 400

@app.route('/upload_resume_for_skill_gap', methods=['POST'])
def upload_resume_for_skill_gap():
    """Handles resume upload for skill gap analysis only."""
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    resume_file = request.files['resume']
    if resume_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if resume_file and resume_file.filename.endswith('.pdf'):
        filename = secure_filename(resume_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(filepath)
        
        resume_text = pdf_to_text(filepath)
        
        if not resume_text:
            os.remove(filepath)
            return jsonify({'error': 'Could not extract text from PDF. The file might be corrupted or image-only.'}), 500

        os.remove(filepath)
        return jsonify({'resume_text': resume_text})

    return jsonify({'error': 'Unsupported file type. Please upload a PDF file.'}), 400

@app.route('/generate_cover_letter', methods=['POST'])
def generate_cover_letter_endpoint():
    """Handles cover letter generation based on job description and resume text."""
    if not request.json or 'job_description' not in request.json or 'resume_text' not in request.json:
        return jsonify({'error': 'Missing job description or resume text in JSON body'}), 400

    job_description = request.json['job_description']
    resume_text = request.json['resume_text']
    template_style = request.json.get('template_style', 'professional')

    if not job_description or not resume_text:
        return jsonify({'error': 'Job description and resume text cannot be empty'}), 400

    # Call Gemini to generate cover letter
    cover_letter_text = generate_cover_letter(job_description, resume_text, template_style)
    
    if cover_letter_text.startswith("Gemini API Error:"):
        return jsonify({'error': cover_letter_text}), 500
        
    return jsonify({'cover_letter': cover_letter_text})

@app.route('/analyze_skill_gap', methods=['POST'])
def analyze_skill_gap():
    """Analyzes skill gaps between resume and job description."""
    if not client:
        return jsonify({'error': 'Gemini Client not initialized.'}), 500

    if not request.json or 'job_description' not in request.json or 'resume_text' not in request.json:
        return jsonify({'error': 'Missing job description or resume text in JSON body'}), 400

    job_description = request.json['job_description']
    resume_text = request.json['resume_text']

    if not job_description or not resume_text:
        return jsonify({'error': 'Job description and resume text cannot be empty'}), 400

    # 1. Define the Skill Gap Analysis Prompt
    system_instruction = (
        "You are an expert career advisor and skills analyst. "
        "Your task is to analyze the given resume against the job description to identify: "
        "1. Skills that match between the resume and job requirements "
        "2. Missing skills and areas for improvement "
        "Your output MUST be a single JSON object that strictly conforms to the provided schema."
    )

    prompt = f"""
    Job Description:\n---\n{job_description}\n---\n
    Resume:\n---\n{resume_text}\n---\n
    
    Analyze the resume against the job description and provide:
    1. A comprehensive list of matching skills (technical skills, soft skills, tools, technologies, etc.)
    2. Missing skills and areas for improvement that would make the candidate more suitable for this role
    
    Focus on specific, actionable insights that help the candidate understand their skill alignment.
    """

    # 2. Define the JSON Response Schema
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "matching_skills": types.Schema(
                type=types.Type.STRING, 
                description="HTML formatted list of skills that match between resume and job description. Use bullet points and highlight key matches."
            ),
            "improvements": types.Schema(
                type=types.Type.STRING, 
                description="HTML formatted list of missing skills and areas for improvement. Use bullet points and be specific about what's needed."
            ),
        },
        required=["matching_skills", "improvements"]
    )

    try:
        gemini_response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.3 # Keep it focused and analytical
            ),
        )

        # The response text is a JSON string, parse it
        skill_analysis = json.loads(gemini_response.text)
        return jsonify(skill_analysis)

    except Exception as e:
        print(f"Skill Gap Analysis API Error: {e}")
        return jsonify({'error': f"Failed to analyze skill gaps: {e}"}), 500

# --- ATS SCORING ROUTE ---
@app.route('/get_ats_score', methods=['POST'])
def get_ats_score():
    """Calculates and returns the general ATS score using Gemini and JSON schema."""
    if not client:
        return jsonify({'error': 'Gemini Client not initialized.'}), 500

    if not request.json or 'original_resume' not in request.json:
        return jsonify({'error': 'Missing original resume input for scoring in JSON body.'}), 400

    original_resume = request.json['original_resume']

    # 1. Define the General ATS Score Prompt
    system_instruction = (
        "You are an expert ATS (Applicant Tracking System) analyst and Senior Recruiter. "
        "Your task is to evaluate the given resume for general ATS compatibility and overall quality. "
        "Provide a numerical score out of 100 that represents the likelihood this resume will pass ATS screening for most job applications. "
        "Focus on technical formatting, keyword optimization, structure, and overall professional presentation. "
        "Your output MUST be a single JSON object that strictly conforms to the provided schema."
    )

    prompt = f"""
    Resume to Analyze:\n---\n{original_resume}\n---\n
    
    Analyze this resume for general ATS compatibility. Consider:
    - Technical formatting (proper headers, consistent structure, clean layout)
    - Keyword optimization and industry-relevant terms
    - Contact information completeness
    - Professional summary quality
    - Skills section organization
    - Work experience formatting and detail
    - Education section completeness
    - Overall readability and ATS-friendly formatting
    - Length appropriateness (not too short, not too long)
    - Action verbs and quantified achievements
    
    Generate a score that reflects how well this resume would perform across various job applications and ATS systems.
    """

    # 2. Define the JSON Response Schema (Structured Output)
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "ats_score": types.Schema(type=types.Type.INTEGER, description="The general ATS compatibility score out of 100. Must be between 0 and 100."),
            "strengths": types.Schema(type=types.Type.STRING, description="List 3-5 key strengths of the resume that make it ATS-friendly."),
            "improvements": types.Schema(type=types.Type.STRING, description="List 3-5 specific areas for improvement to increase ATS compatibility."),
            "overall_assessment": types.Schema(type=types.Type.STRING, description="A brief overall assessment of the resume's ATS readiness and professional quality."),
        },
        required=["ats_score", "strengths", "improvements", "overall_assessment"]
    )

    try:
        gemini_response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.2 # Keep it objective/deterministic
            ),
        )

        # The response text is a JSON string, parse it
        scorecard = json.loads(gemini_response.text)
        return jsonify(scorecard)

    except Exception as e:
        print(f"ATS Score API Error: {e}")
        return jsonify({'error': f"Failed to generate ATS score: {e}"}), 500

# --- CHAT BOT ROUTE (Integration of your chat logic) ---
@app.route('/chat', methods=['POST'])
def chat():
    """Conversational endpoint to assist with resume editing and advice."""
    if not client:
        return jsonify({'error': 'Gemini Client not initialized.'}), 500

    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    body = request.get_json(silent=True) or {}
    user_message = body.get('message', '').strip()
    job_description = body.get('job_description', '').strip()
    # The current preview is sent as HTML from the editable box
    current_preview = body.get('current_preview', '').strip()

    if not user_message:
        return jsonify({'error': 'Missing message'}), 400

    # System instruction for the chat model
    system_instruction = (
        "You are a friendly, concise resume and cover letter writing copilot for end users. "
        "Help improve the user's resume or cover letter for a given job description. "
        "When the user asks for a change, produce the updated full text (resume or cover letter), formatted with clear Markdown (bold for sections, bullet points). "
        "For cover letters, maintain proper business letter formatting with appropriate spacing and structure. "
        "If the user is only asking for advice, set 'updated_preview' to an empty string. "
        "Keep the tone professional and helpful. "
        "Determine from context whether the user is editing a resume or cover letter."
    )

    # Structured output for predictable response handling in the frontend
    response_schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "reply_text": types.Schema(type=types.Type.STRING, description="Short assistant reply to display in chat."),
            "updated_preview": types.Schema(type=types.Type.STRING, description="If the user requested edits, the fully updated resume text in Markdown. Otherwise empty string."),
            "reasoning_summary": types.Schema(type=types.Type.STRING, description="One or two concise sentences that explain the assistant's approach."),
            "deliberation_steps": types.Schema(type=types.Type.INTEGER, description="Rough step-count the model considered (1-10)."),
        },
        required=["reply_text", "updated_preview", "reasoning_summary", "deliberation_steps"]
    )

    prompt_parts = []
    if job_description:
        prompt_parts.append(f"Job Description:\n---\n{job_description}\n---\n")
    if current_preview:
        # Check if this is a cover letter or resume based on prefix
        if current_preview.startswith("COVER_LETTER:"):
            content = current_preview.replace("COVER_LETTER:", "")
            prompt_parts.append(f"Current Cover Letter Preview (Raw HTML/Text):\n---\n{content}\n---\n")
        elif current_preview.startswith("RESUME:"):
            content = current_preview.replace("RESUME:", "")
            prompt_parts.append(f"Current Resume Preview (Raw HTML/Text):\n---\n{content}\n---\n")
        else:
            # Legacy support for old format
            prompt_parts.append(f"Current Resume Preview (Raw HTML/Text):\n---\n{current_preview}\n---\n")
    prompt_parts.append(f"User Message:\n---\n{user_message}\n---\n")
    prompt_parts.append(
        "If the user's message requests modifications, update the document fully and return it as updated_preview in Markdown format. "
        "Prefix your response with 'COVER_LETTER:' if editing a cover letter, or 'RESUME:' if editing a resume. "
        "If no update is needed, set updated_preview to an empty string. "
        "Do not include any introductory dialogue in the 'updated_preview' field."
    )

    try:
        gemini_response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents="\n\n".join(prompt_parts),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.4
            ),
        )

        data = json.loads(gemini_response.text)
        
        # Prepare response data, using 'None' for empty strings to simplify frontend logic
        reply_text = (data.get('reply_text') or '').strip()
        updated_preview = (data.get('updated_preview') or '').strip()
        reasoning_summary = (data.get('reasoning_summary') or '').strip()
        deliberation_steps = data.get('deliberation_steps')

        return jsonify({
            'reply_text': reply_text,
            'updated_preview': updated_preview or None, # Use None for empty strings
            'reasoning_summary': reasoning_summary or None,
            'deliberation_steps': deliberation_steps if isinstance(deliberation_steps, int) else None
        })
    except Exception as e:
        print(f"Chat API Error: {e}")
        return jsonify({'error': f"Failed to process chat: {e}"}), 500

if __name__ == '__main__':
    # You will use a file named .env to store your GEMINI_API_KEY
    if not os.getenv("GEMINI_API_KEY"):
         print("WARNING: GEMINI_API_KEY environment variable not set. Please create a .env file with your key.")
    # IMPORTANT: Flask will look for 'index.html' in a folder named 'templates'
    app.run(debug=True)
import os
import json
import hashlib
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document


# --- OCR Imports (Safe Mode) ---
try:
    from PIL import Image
    import pytesseract
    # IF TESSERACT IS INSTALLED BUT NOT FOUND, UNCOMMENT AND UPDATE THIS PATH:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    Image = None
    pytesseract = None

# --- GEMINI & LANGCHAIN IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# ================= CONFIG =================

# !!! PASTE YOUR GOOGLE API KEY HERE !!!
os.environ["GOOGLE_API_KEY"] = "AIzaSyAtnDl7QmUa8zKSd6pyOAPizQdaxk_w8oQ" 

RESUME_DIR = "resumes"
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "processed_data.csv")
JSON_PATH = os.path.join(DATA_DIR, "processed_files.json")

os.makedirs(RESUME_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Gemini
try:
    # We use gemini-1.5-flash for speed and low cost (free tier available)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        convert_system_message_to_human=True # Helps with older LangChain versions
    )
except Exception as e:
    st.error(f"Gemini Error: {e}")
    llm = None

# ================= SCHEMA =================

class ResumeSchema(BaseModel):
    name: str = Field(description="Candidate full name")
    years_of_experience: float = Field(description="Total years of experience (numeric)")
    skills: list[str] = Field(description="List of technical skills")
    degree: str = Field(description="Highest degree obtained")

parser = JsonOutputParser(pydantic_object=ResumeSchema)

# Gemini sometimes needs a stricter prompt to ensure JSON
PROMPT = PromptTemplate(
    template="""
    You are an expert ATS (Applicant Tracking System).
    Analyze the resume text below and extract structured data.
    
    IMPORTANT: Return ONLY raw JSON. Do not use Markdown formatting (like ```json).
    
    {format_instructions}
    
    Resume Text:
    {text}
    """,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# ================= HELPERS =================

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def load_json():
    if not os.path.exists(JSON_PATH): return {}
    with open(JSON_PATH, "r") as f: return json.load(f)

def save_json(data):
    with open(JSON_PATH, "w") as f: json.dump(data, f, indent=2)

def extract_text(path):
    ext = path.lower()
    try:
        if ext.endswith(".pdf"):
            reader = PdfReader(path)
            text = ""
            for p in reader.pages:
                extracted = p.extract_text()
                if extracted: text += extracted + " "
            return text

        if ext.endswith(".docx"):
            doc = Document(path)
            return " ".join(p.text for p in doc.paragraphs)

        if ext.endswith((".png", ".jpg", ".jpeg")):
            if pytesseract and Image:
                img = Image.open(path)
                return pytesseract.image_to_string(img)
            else:
                return "[OCR Library Missing]"

    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""
    return ""

def analyze_resume(text):
    if not llm: return None
    chain = PROMPT | llm | parser
    try:
        return chain.invoke({"text": text[:5000]}) # Gemini has a larger context window, we can send more text!
    except Exception as e:
        st.error(f"Gemini Extraction Failed: {e}")
        return None

# ================= UI =================

st.set_page_config("Gemini Resume Shortlister", layout="wide")
st.title("📄 AI Resume Shortlister (Powered by Gemini)")

processed_files = load_json()
existing_df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()

# ================= PROCESS =================

if st.button("🚀 Process New Resumes"):
    if not os.path.exists(RESUME_DIR):
        st.error(f"Folder '{RESUME_DIR}' not found.")
    else:
        rows = []
        files = os.listdir(RESUME_DIR)
        progress = st.progress(0)
        status = st.empty()

        for i, file in enumerate(files):
            path = os.path.join(RESUME_DIR, file)
            if not os.path.isfile(path): continue

            h = file_hash(path)
            if h in processed_files: continue

            status.text(f"Asking Gemini about {file}...")
            text = extract_text(path)
            
            if not text.strip():
                st.warning(f"Skipped {file} (Empty or Scanned)")
                continue

            data = analyze_resume(text)
            if data:
                data["filename"] = file
                data["filepath"] = os.path.abspath(path)
                rows.append(data)
                processed_files[h] = {"filename": file, "path": os.path.abspath(path)}
            
            progress.progress((i + 1) / len(files))

        status.empty()

        if rows:
            new_df = pd.DataFrame(rows)
            final_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
            final_df.to_csv(CSV_PATH, index=False)
            save_json(processed_files)
            st.success(f"✅ Processed {len(rows)} new resumes!")
            st.rerun()
        else:
            st.info("No new resumes to process.")

# ================= SEARCH LOGIC (SMART HYBRID) =================

if os.path.exists(CSV_PATH):
    st.divider()
    
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    
    # --- FIX 1: Convert Experience to Numbers ---
    # Force 'years_of_experience' to be numeric, turning errors (like "N/A") into 0
    df['years_of_experience'] = pd.to_numeric(df['years_of_experience'], errors='coerce').fillna(0)

    # --- FIX 2: Clean Skills List ---
    import ast
    def clean_skills(skill_str):
        try:
            return ast.literal_eval(skill_str) if isinstance(skill_str, str) else []
        except:
            return []
            
    df['skills_list'] = df['skills'].apply(clean_skills)

    with st.expander(f"📊 View Database ({len(df)} candidates)"):
        st.dataframe(df, use_container_width=True)

    query = st.text_input("Describe your ideal candidate (e.g., 'Python and Java dev with 3+ years exp')")

    if query and llm:
        # STEP 1: Use Gemini to Convert Natural Language -> Structured Rules
        rule_prompt = f"""
        You are a smart recruiter. Analyze this hiring query: "{query}"
        
        Extract these 2 criteria:
        1. required_skills: A list of MUST-HAVE skills (lowercase).
        2. min_experience: Minimum years of experience (number). Default to 0 if not mentioned.
        
        Return ONLY valid JSON. Format:
        {{"required_skills": ["python", "java"], "min_experience": 3}}
        """
        
        with st.spinner("Analyzing requirements..."):
            try:
                response = llm.invoke(rule_prompt).content
                response = response.replace("```json", "").replace("```", "").strip()
                criteria = json.loads(response)
                
                req_skills = [s.lower() for s in criteria.get("required_skills", [])]
                min_exp = criteria.get("min_experience", 0)
                
                st.info(f"🔍 Filtering for: **Skills:** {req_skills} | **Min Exp:** {min_exp}+ years")
                
                # STEP 2: Use Python to Filter (Deterministic & Complete)
                matches = []
                
                for index, row in df.iterrows():
                    # Check Experience (Now safe because we converted to numeric above)
                    if row['years_of_experience'] < min_exp:
                        continue
                        
                    # Check Skills (Candidate must have ALL required skills)
                    cand_skills = [str(s).lower() for s in row['skills_list']]
                    
                    # Logic: Are all required skills present in candidate skills?
                    if all(req in cand_skills for req in req_skills):
                        matches.append(row)

                # STEP 3: Display Results
                if matches:
                    st.success(f"✅ Found {len(matches)} Matching Candidates")
                    
                    for row in matches:
                        with st.container(border=True):
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"### {row['name']}")
                                st.markdown(f"**Exp:** {row['years_of_experience']} Yrs | **Degree:** {row['degree']}")
                                st.caption(f"Skills: {', '.join(map(str, row['skills_list']))}")
                            with c2:
                                # DYNAMIC PATH RECONSTRUCTION (Fixes "File not found")
                                safe_path = os.path.join(RESUME_DIR, row['filename'])
                                
                                if os.path.exists(safe_path):
                                    with open(safe_path, "rb") as f:
                                        st.download_button(
                                            label="📂 Open Resume",
                                            data=f,
                                            file_name=row['filename'],
                                            mime="application/pdf",
                                            key=f"btn_{row['filename']}"
                                        )
                                else:
                                    st.warning("File missing")
                else:
                    st.warning("No candidates met all the criteria.")
                    
            except Exception as e:
                st.error(f"Search Logic Error: {e}")
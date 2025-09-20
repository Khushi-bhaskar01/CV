import os
import re
import json
import requests
from pypdf import PdfReader
import io
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserIdRequest(BaseModel):
    user_id: str

class CVAnalysisResponse(BaseModel):
    success: bool
    message: str
    user_id: str
    resume_url: str = None
    analysis: Dict[str, Any] = None
    stored_in_firestore: bool = False
    error_details: str = None

# Initialize Firebase once when the module loads
def initialize_firebase():
    """Initialize Firebase Admin SDK using environment variable"""
    try:
        # Check if Firebase is already initialized
        try:
            firebase_admin.get_app()
            print("Firebase already initialized")
            return True
        except ValueError:
            # Firebase not initialized, so initialize it
            pass
        
        # Get Firebase credentials from environment variable
        firebase_creds = os.environ.get("FIREBASE_CREDENTIALS_JSON")
        if not firebase_creds:
            raise Exception("FIREBASE_CREDENTIALS_JSON environment variable not found")
        
        # Parse the JSON credentials
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        print("Firebase initialized successfully from environment variable")
        return True
        
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return False

# Initialize Firebase when the module loads
firebase_initialized = initialize_firebase()
if firebase_initialized:
    db = firestore.client()
else:
    db = None

class ResumeAnalysisSystem:
    """Streamlined system to analyze resumes from Firestore with genuine ATS scoring"""
    
    def __init__(self):
        """Initialize the system with API keys"""
        load_dotenv()
        
        # Get API keys from environment variables
        openai_key = os.environ.get("OPENAI_API_KEY")
        groq_key = os.environ.get("GROQ_API_KEY")
        
        if not openai_key:
            raise Exception("OPENAI_API_KEY not found in environment variables")
        if not groq_key:
            raise Exception("GROQ_API_KEY not found in environment variables")
            
        os.environ["OPENAI_API_KEY"] = openai_key
        os.environ["GROQ_API_KEY"] = groq_key
        
        if not db:
            raise Exception("Firebase not initialized")
        
        self.db = db
        print("Resume Analysis System initialized")

    def get_resume_urls(self, user_id):
        """Retrieve resume URLs for a specific user ID from Firestore"""
        try:
            query = self.db.collection("resumes").where(
                filter=FieldFilter("uid", "==", user_id)
            )
            docs = list(query.stream())
            
            urls = []
            for doc in docs:
                doc_data = doc.to_dict()
                if "resumeFile" in doc_data and isinstance(doc_data["resumeFile"], dict):
                    if "url" in doc_data["resumeFile"]:
                        url = doc_data["resumeFile"]["url"]
                        if url:
                            urls.append(url)
            
            return urls
            
        except Exception as e:
            print(f"Error retrieving URLs: {e}")
            return []

    def analyze_resume_from_url(self, pdf_url, user_query=None):
        """Analyze a resume PDF from URL using AI for genuine ATS assessment"""
        try:
            # Default query focused on comprehensive ATS analysis
            if user_query is None:
                user_query = """
                Please perform a comprehensive ATS (Applicant Tracking System) analysis of this resume.

                Analyze the following aspects and provide scores based on actual content:

                1. ATS COMPATIBILITY SCORE (0-100):
                - Keyword density and relevance
                - Section headers and structure
                - File format compatibility
                - Text readability by ATS systems
                - Use of standard formatting

                2. CONTENT QUALITY SCORE (0-100):
                - Relevance of experience
                - Skills alignment with industry standards
                - Achievement quantification
                - Professional language usage
                - Completeness of information

                3. FORMATTING SCORE (0-100):
                - Clean, professional layout
                - Consistent formatting
                - Appropriate use of white space
                - Font and size consistency
                - Section organization

                Provide your response in this exact format:

                ATS Score: [numerical score 0-100]
                Content Score: [numerical score 0-100]
                Formatting Score: [numerical score 0-100]

                Keywords Found:
                - [list relevant keywords found]

                Keywords Missing:
                - [list important missing keywords for this field]

                IMPROVEMENT SUGGESTIONS:
                1. **[Title]**: [Specific actionable improvement]
                2. **[Title]**: [Specific actionable improvement]
                3. **[Title]**: [Specific actionable improvement]
                4. **[Title]**: [Specific actionable improvement]
                5. **[Title]**: [Specific actionable improvement]

                Base all scores on actual resume content analysis, not assumptions.
                """
            
            print("Fetching and processing PDF...")
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Process PDF content using pypdf
            pdf_reader = PdfReader(io.BytesIO(response.content))
            docs = []
            full_text = ""
            
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                full_text += text + "\n"
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_url, "page": i + 1}
                    )
                )
            
            # Create embeddings and vector store
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split = text_splitter.split_documents(docs)
            
            embed = OpenAIEmbeddings(model="text-embedding-3-large")
            vector_db = FAISS.from_documents(split, embed)
            
            # Initialize LLM
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            
            # System prompt for professional ATS analysis
            system_prompt = """
            You are an expert ATS (Applicant Tracking System) analyzer and professional recruiter.
            Your job is to evaluate resumes exactly as modern ATS systems would, considering:

            1. Keyword matching and density
            2. Section recognition and parsing
            3. Format compatibility
            4. Information completeness
            5. Professional standards

            Provide honest, data-driven scores based on the actual content provided.
            Be specific about what you observe in the resume.
            All scores should reflect genuine assessment, not arbitrary numbers.
            """
            
            # Get comprehensive analysis
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Resume Content:\n{full_text}\n\nAnalysis Request:\n{user_query}")
            ]
            
            result = llm.invoke(messages)
            return result.content
            
        except Exception as e:
            print(f"Error analyzing resume: {e}")
            return f"Error analyzing resume: {e}"

    def debug_analysis_response(self, analysis_text, user_id):
        """Debug method to understand the LLM response format"""
        print(f"\n=== DEBUGGING ANALYSIS RESPONSE FOR USER {user_id} ===")
        print(f"Response length: {len(analysis_text)} characters")
        print("\n--- First 500 characters ---")
        print(analysis_text[:500])
        print("\n--- Last 500 characters ---")
        print(analysis_text[-500:])
        
        # Look for suggestion markers
        suggestion_markers = ['IMPROVEMENT', 'SUGGESTIONS', 'suggestion', 'improve']
        for marker in suggestion_markers:
            positions = [i for i, word in enumerate(analysis_text.lower().split()) if marker in word.lower()]
            if positions:
                print(f"\nFound '{marker}' at word positions: {positions}")
        
        # Check for numbered items
        numbered_items = re.findall(r'\n\s*\d+\.\s*[^\n]+', analysis_text)
        print(f"\nNumbered items found: {len(numbered_items)}")
        for item in numbered_items[:5]:
            print(f"  {item.strip()}")
        
        # Check for bullet points
        bullet_items = re.findall(r'\n\s*[-•*]\s*[^\n]+', analysis_text)
        print(f"\nBullet items found: {len(bullet_items)}")
        for item in bullet_items[:5]:
            print(f"  {item.strip()}")
        
        print("\n=== END DEBUG ===\n")

    def parse_analysis_response(self, analysis_text):
        """Fixed parsing method that correctly extracts suggestions from your specific format"""
        try:
            # Initialize structure
            parsed_data = {
                "Analysis": {
                    "atsScore": "",
                    "contentScore": "",
                    "formattingScore": "",
                    "keywordsFound": [],
                    "keywordsMissing": []
                },
                "suggestions": []
            }
            
            # Extract scores - keeping your existing logic as it works
            patterns = {
                "ats": [
                    r"ATS\s*Score[:\s]*(\d+(?:\.\d+)?)",
                    r"ATS[:\s]*(\d+(?:\.\d+)?)"
                ],
                "content": [
                    r"Content\s*(?:Quality\s*)?Score[:\s]*(\d+(?:\.\d+)?)",
                    r"Content[:\s]*(\d+(?:\.\d+)?)"
                ],
                "formatting": [
                    r"Formatting\s*Score[:\s]*(\d+(?:\.\d+)?)",
                    r"Formatting[:\s]*(\d+(?:\.\d+)?)"
                ]
            }
            
            # Extract scores
            for score_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, analysis_text, re.IGNORECASE)
                    if match:
                        score = str(int(float(match.group(1))))
                        if score_type == "ats":
                            parsed_data["Analysis"]["atsScore"] = score
                        elif score_type == "content":
                            parsed_data["Analysis"]["contentScore"] = score
                        elif score_type == "formatting":
                            parsed_data["Analysis"]["formattingScore"] = score
                        break
            
            # Extract keywords found - Fixed pattern
            keywords_found_section = re.search(
                r'\*\*Keywords?\s*Found:\*\*\s*\n(.*?)(?=\*\*Keywords?\s*Missing:|\*\*IMPROVEMENT|\Z)', 
                analysis_text, 
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            )
            if keywords_found_section:
                keywords_text = keywords_found_section.group(1)
                keywords = re.findall(r'^[-•*]\s*(.+)$', keywords_text, re.MULTILINE)
                parsed_data["Analysis"]["keywordsFound"] = [kw.strip() for kw in keywords if kw.strip()]
            
            # Extract keywords missing - Fixed pattern
            keywords_missing_section = re.search(
                r'\*\*Keywords?\s*Missing:\*\*\s*\n(.*?)(?=\*\*IMPROVEMENT|\Z)', 
                analysis_text, 
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            )
            if keywords_missing_section:
                keywords_text = keywords_missing_section.group(1)
                keywords = re.findall(r'^[-•*]\s*(.+)$', keywords_text, re.MULTILINE)
                parsed_data["Analysis"]["keywordsMissing"] = [kw.strip() for kw in keywords if kw.strip()]
            
            # Extract suggestions - COMPLETELY REWRITTEN for your format
            improvement_section = re.search(
                r'\*\*IMPROVEMENT\s*SUGGESTIONS:\*\*\s*\n(.*?)(?=\n\n|\Z)', 
                analysis_text, 
                re.IGNORECASE | re.MULTILINE | re.DOTALL
            )
            
            if improvement_section:
                suggestions_text = improvement_section.group(1)
                print(f"Found improvement section: {len(suggestions_text)} characters")
                
                # Pattern to match: "1. **Title**: Description"
                suggestion_pattern = r'(\d+)\.\s*\*\*([^*]+)\*\*:\s*([^\n]+(?:\n(?!\d+\.)[^\n]*)*)'
                matches = re.findall(suggestion_pattern, suggestions_text, re.MULTILINE)
                
                print(f"Pattern matches found: {len(matches)}")
                
                for i, (number, title, description) in enumerate(matches):
                    suggestion_data = {
                        "id": str(i + 1),
                        "category": self._categorize_suggestion(title + " " + description),
                        "title": title.strip(),
                        "description": description.strip(),
                        "severity": self._determine_severity(description)
                    }
                    parsed_data["suggestions"].append(suggestion_data)
                    print(f"Added suggestion {i+1}: {title[:30]}...")
            
            # Fallback method if the above didn't work
            if not parsed_data["suggestions"]:
                print("Primary extraction failed, trying fallback method...")
                
                fallback_pattern = r'SUGGESTIONS?.*?\n(.*?)(?:\n\n|\Z)'
                fallback_match = re.search(fallback_pattern, analysis_text, re.IGNORECASE | re.DOTALL)
                
                if fallback_match:
                    suggestions_section = fallback_match.group(1)
                    numbered_items = re.findall(r'(\d+)\.\s*(.+?)(?=\n\d+\.|\Z)', suggestions_section, re.DOTALL)
                    
                    for i, (number, content) in enumerate(numbered_items):
                        content = re.sub(r'\n+', ' ', content).strip()
                        
                        title_match = re.search(r'\*\*([^*]+)\*\*:\s*(.*)', content)
                        if title_match:
                            title = title_match.group(1).strip()
                            description = title_match.group(2).strip()
                        else:
                            parts = content.split(':', 1)
                            title = parts[0][:50] + "..." if len(parts[0]) > 50 else parts[0]
                            description = parts[1] if len(parts) > 1 else content
                        
                        suggestion_data = {
                            "id": str(i + 1),
                            "category": self._categorize_suggestion(content),
                            "title": title,
                            "description": description.strip(),
                            "severity": self._determine_severity(content)
                        }
                        parsed_data["suggestions"].append(suggestion_data)
                        print(f"Fallback added suggestion {i+1}: {title[:30]}...")
            
            print(f"Final result: {len(parsed_data['suggestions'])} suggestions extracted")
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing analysis: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "Analysis": {
                    "atsScore": "0",
                    "contentScore": "0", 
                    "formattingScore": "0",
                    "keywordsFound": [],
                    "keywordsMissing": []
                },
                "suggestions": [{
                    "id": "1",
                    "category": "Error",
                    "title": "Parsing Error",
                    "description": f"Failed to parse suggestions: {str(e)}",
                    "severity": "high"
                }]
            }

    def _categorize_suggestion(self, suggestion_text):
        """Categorize suggestion based on content"""
        suggestion_lower = suggestion_text.lower()
        
        if any(word in suggestion_lower for word in ['keyword', 'skill', 'technology', 'certification']):
            return "Keywords"
        elif any(word in suggestion_lower for word in ['format', 'layout', 'structure', 'design']):
            return "Formatting" 
        elif any(word in suggestion_lower for word in ['experience', 'achievement', 'responsibility']):
            return "Content"
        elif any(word in suggestion_lower for word in ['contact', 'email', 'phone', 'linkedin']):
            return "Contact"
        else:
            return "General"

    def _determine_severity(self, suggestion_text):
        """Determine suggestion severity based on language used"""
        suggestion_lower = suggestion_text.lower()
        
        if any(word in suggestion_lower for word in ['critical', 'must', 'essential', 'required']):
            return "high"
        elif any(word in suggestion_lower for word in ['should', 'important', 'recommend']):
            return "medium"
        else:
            return "low"

    def store_analysis_in_firestore(self, user_id, analysis_data):
        """Store the parsed analysis data in Firestore"""
        try:
            user_ref = self.db.collection("user").document(user_id)
            user_ref.update({"resume": analysis_data})
            print("Analysis stored successfully in Firestore")
            return True
        except Exception as e:
            print(f"Error storing analysis: {e}")
            return False

    def analyze_user_resumes_complete(self, user_id, store_results=True):
        """Complete workflow: analyze user's resume and optionally store results"""
        print(f"Analyzing resumes for user: {user_id}")
        
        # Get resume URLs
        urls = self.get_resume_urls(user_id)
        if not urls:
            raise Exception(f"No resumes found for user {user_id}")
        
        # Analyze first resume
        analysis_text = self.analyze_resume_from_url(urls[0])
        
        # Debug the response to understand format issues
        self.debug_analysis_response(analysis_text, user_id)
        
        parsed_analysis = self.parse_analysis_response(analysis_text)
        
        # Store if requested
        storage_success = False
        if store_results:
            storage_success = self.store_analysis_in_firestore(user_id, parsed_analysis)
        
        return {
            "user_id": user_id,
            "resume_url": urls[0],
            "analysis": parsed_analysis,
            "stored_in_firestore": storage_success
        }

def process_cv_analysis(user_id: str) -> Dict[str, Any]:
    """Main function to process CV analysis for a user"""
    try:
        if not db:
            raise Exception("Firebase not initialized")
        
        print(f"Processing CV analysis for user_id: {user_id}")
        
        # Create resume analysis system instance
        resume_system = ResumeAnalysisSystem()
        
        # Analyze user's resume
        result = resume_system.analyze_user_resumes_complete(user_id, store_results=True)
        
        return {
            "success": True,
            "message": "CV analysis completed successfully",
            **result
        }
        
    except Exception as e:
        print(f"Error processing CV analysis for user {user_id}: {str(e)}")
        return {
            "success": False,
            "message": "Failed to process CV analysis",
            "user_id": user_id,
            "error_details": str(e)
        }

@app.post("/analyze-cv", response_model=CVAnalysisResponse)
async def analyze_cv(request: UserIdRequest):
    """Endpoint to analyze user's CV/resume"""
    try:
        if not firebase_initialized:
            raise HTTPException(status_code=500, detail="Firebase initialization failed")
        
        if not request.user_id or not request.user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty")
        
        print(f"Received request to analyze CV for user_id: {request.user_id}")
        
        # Process CV analysis
        result = process_cv_analysis(request.user_id.strip())
        
        if result["success"]:
            return CVAnalysisResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result["error_details"])
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in CV analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "firebase_initialized": firebase_initialized,
        "message": "CV Analysis Service is running",
        "available_endpoints": [
            "POST /analyze-cv - Analyze user's CV/resume",
            "GET /health - Health check"
        ]
    }

# For Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

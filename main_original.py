import gradio as gr
import os 
import json
from typing import Tuple, List, Optional
from pypdf import PdfReader 
from dotenv import load_dotenv
from openai import OpenAI # The Google AI SDK uses the OpenAI client structure
# from weasyprint import HTML

# --- Configuration ---
DEBUG_MODE = False 

# Global state to hold the instance of the AI Assistant class
assistant_instance = None

# --- AI Assistant Class ---

class AIAssistant:
    """
    Manages the Gemini tool-using agent logic, conversation history, 
    and the resume/design tool calls.
    """
    def __init__(self, resume_content: str, job_content: str):
        # 1. Load environment variables
        load_dotenv(override=True)
        self.GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        
        # Base URL for Gemini models via the OpenAI client
        self.GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.MODEL = "gemini-2.5-flash-preview-05-20"
        
        # 2. Initialize the AI client
        self.coach_ai = OpenAI(base_url=self.GEMINI_BASE_URL, api_key=self.GOOGLE_API_KEY)

        # 3. Store the user inputs (Original content)
        self.resume_detail = resume_content 
        self.job_detail = job_content      
        
        # 4. Define the Coach Prompt
        self.coach_prompt = (
            "You are a supportive and experienced career coach who specializes in tailoring resumes to match job descriptions effectively. "
            "Your role is to guide the candidate in creating a compelling and confident resume summary that aligns closely with the job they are applying for.\n\n"
            "You will be provided with the candidate's current resume and the job description. "
            "After reviewing them, you may ask the candidate a few short, focused questions to gather any missing or clarifying details. "
            "Once you have enough information, conclude the conversation by thanking the candidate. "
            "After that, call the 'resume_wrapper' tool to generate the optimized resume.\n\n"
            f"Current Resume (Snippet):\n{self.resume_detail[:500]}...\n\n"
            f"Job Description (Snippet):\n{self.job_detail[:500]}...\n\n"
        )
        
        # 5. Define Tools for the Agent
        resume_wrapper_json = {
            "name": "resume_wrapper",
            "description": "Use this tool to generate the final optimized resume as an HTML file. Call this only once you have enough information and are ready to conclude the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_summary": {
                        "type": "string",
                        "description": "A concise summary (max 300 words) of the candidate's profile, including refinements made during the chat, to be used for the resume generation."
                    }
                },
                "required": ["final_summary"],
                "additionalProperties": False
            }
        }
        self.tools = [{"type": "function", "function": resume_wrapper_json}]
        
        # 6. Generate the initial opening message
        messages = [{"role": "user", "content": self.coach_prompt + "Please start the conversation."}]
        coach_opening_response = self.coach_ai.chat.completions.create(model=self.MODEL, messages=messages).choices[0].message.content
        
        # 7. Set the initial chat history (for the agent's context)
        self.open_chat = [
            {"role": "system", "content": self.coach_prompt},
            {"role": "assistant", "content": coach_opening_response}
        ]
        
    # --- Tool Functions (Agent Backend) ---

    def resume_writing(self, final_summary: str) -> str:
        """Generates the optimized resume content in JSON format."""
        writer_prompt = (
            "You are an expert resume writer. You will be given three inputs: "
            "1) the candidate's **original** resume, 2) the job description, and 3) the **final summary** based on the coaching session. "
            "Your task is to generate an optimized resume in **JSON format only** (no markdown, no code blocks). "
            "The JSON must incorporate the feedback from the final summary and align with the job description.\n\n"
            f"Candidate Summary:\n{final_summary}\n\n" 
            f"Original Resume:\n{self.resume_detail}\n\n"
            f"Job Description:\n{self.job_detail}\n\n"
        )
        messages = [{"role": "user", "content": writer_prompt}]
        response = self.coach_ai.chat.completions.create(model=self.MODEL, messages=messages).choices[0].message.content
        return response

    def resume_design(self, resume_json: str) -> str:
        """Converts the resume JSON into a clean, professional HTML layout."""
        design_prompt = (
            "You are an expert in resume design. You will receive a resume in JSON format, "
            "and your task is to convert it into a clean, professional HTML layout. "
            "The HTML should be properly structured and styled for later conversion into an A4 PDF document. "
            "Do not include markdown formatting or code blocks—return raw HTML only.\n\n"
            f"Resume JSON:\n{resume_json}\n\n"
        )
        messages = [{"role": "user", "content": design_prompt}]
        response = self.coach_ai.chat.completions.create(model=self.MODEL, messages=messages).choices[0].message.content
        return response

    def resume_wrapper(self, final_summary: str) -> dict:
        """Wraps the generation and design tools, saves the file, and prepares for download."""
        print("Resume_wrapper: Starting generation and design...")
        
        # 1. Generate JSON resume using the final summary and original docs
        resume_answer = self.resume_writing(final_summary) 
        
        # 2. Convert JSON to HTML design
        print("Resume_wrapper: Resume content generated. Starting HTML design...")
        design_answer = self.resume_design(resume_answer)
        
        # 3. Save the file
        output_filepath = "your_new_resume.html"
        with open(output_filepath, "w", encoding="utf-8") as file:
            file.write(design_answer)
            
        # print("Resume_wrapper: Resume content generated. Converting to PDF ...")
        # output_filepath = "your_new_resume.pdf"
        # HTML(string=design_answer).write_pdf(output_filepath)
            
        print(f"Resume saved to {output_filepath}")

        # Return the file path for Gradio to handle the download
        return {"recorded": "ok", "file_path": output_filepath}


    def handle_tool_calls(self, tool_calls: list) -> list:
        """Executes tool calls from the model and returns the results."""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name} with args: {arguments}", flush=True)
            
            tool_method = getattr(self, tool_name, None)
            
            if tool_method:
                # Pass the arguments from the model directly to the class method
                result = tool_method(**arguments)
            else:
                result = {"error": f"Tool {tool_name} not found."}
                
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def chat(self, message: str, history: List[List[Optional[str]]]) -> Tuple[str, Optional[str]]:
        """Handles the multi-turn chat with the model, including tool-use logic."""
        
        # Format Gradio history for the AI client (user/assistant roles)
        messages = self.open_chat.copy()
        
        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
                
        # Append the current user message
        messages.append({"role": "user", "content": message})
        
        done = False
        final_reply = ""
        file_path_found = None 
        
        while not done:
            response = self.coach_ai.chat.completions.create(
                model=self.MODEL,
                messages=messages,
                tools=self.tools
            )
            
            message_content = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == "tool_calls":
                tool_calls = message_content.tool_calls
                results = self.handle_tool_calls(tool_calls)
                
                # Check results for file path
                for result in results:
                    try:
                        data = json.loads(result.get("content", "{}"))
                        if 'file_path' in data:
                            file_path_found = data['file_path'] # Capture the file path
                    except json.JSONDecodeError:
                        continue 
                        
                messages.append(message_content)
                messages.extend(results)
                
            else:
                final_reply = message_content.content
                done = True
        
        # Return the final reply AND the file path
        return final_reply, file_path_found 


# ----------------------------------------------------------------------
# --- GRADIO INTERFACE FUNCTIONS ---
# ----------------------------------------------------------------------

def get_file_content(file_path: Optional[str], text_input: str, debug_mode: bool) -> Tuple[str, str, Optional[str]]:
    """Reads content from file (including PDF extraction) or text input."""
    content = ""
    debug_msg = ""
    source_path = None

    if text_input:
        content = text_input
        debug_msg = f"Content from **Pasted Text**. Length: {len(content)}"
        source_path = "Pasted Text"
    
    elif file_path is not None:
        source_path = file_path
        
        if source_path.lower().endswith(('.pdf', '.txt')):
            try:
                if source_path.lower().endswith('.pdf'): 
                    reader = PdfReader(source_path)
                    # Extract text, joining with a newline, skipping empty pages
                    pdf_content = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())
                    content = pdf_content if pdf_content else f"--- PDF extracted, but no readable text found: {os.path.basename(source_path)} ---"

                elif source_path.lower().endswith(('.txt')):
                    # NOTE: .doc/.docx files are read as text; for full compatibility, textract is needed.
                    with open(source_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                if debug_mode:
                    file_size = os.path.getsize(source_path)
                    content_snippet = content[:150].replace('\n', ' ') 
                    debug_msg = f"File: **{os.path.basename(source_path)}**, Size: {file_size/1024:.2f} KB. Snippet: '{content_snippet}...' "

            except Exception as e:
                content = f"Error reading file {os.path.basename(source_path)}: {e}"
                debug_msg = f"Error processing file: {e}"
        else:
            content = f"Unsupported file type for {os.path.basename(source_path)}."
            debug_msg = "Unsupported file type."

    return content, debug_msg, source_path

def activate_chat(
    resume_file: Optional[str], 
    job_file: Optional[str], 
    resume_text: str, 
    job_text: str, 
    debug_mode: bool
) -> Tuple[gr.update, gr.update, gr.update, gr.update]:
    """Initializes the AI Assistant and reveals the chat interface."""
    global assistant_instance

    resume_content, resume_debug_msg, _ = get_file_content(resume_file, resume_text, debug_mode)
    job_content, job_debug_msg, _ = get_file_content(job_file, job_text, debug_mode)

    is_resume_ready = bool(resume_content) and "Error reading file" not in resume_content
    is_job_ready = bool(job_content) and "Error reading file" not in job_content

    if is_resume_ready and is_job_ready:
        try:
            # Initialize the AI Assistant with the extracted content
            assistant_instance = AIAssistant(resume_content, job_content)
            
            # Extract the initial welcome message from the assistant
            initial_history_content = assistant_instance.open_chat[-1]["content"]

            # Initialize history in the required list of [user, assistant] format
            initial_history: List[List[Optional[str]]] = [
                [None, initial_history_content]
            ]

            debug_output = ""
            if debug_mode:
                debug_output = f"--- Debug Mode Active ---\n\n"
                debug_output += f"Current Resume Debug:\n{resume_debug_msg}\n\n"
                debug_output += f"Job Description Debug:\n{job_debug_msg}"
            
            return (
                gr.update(visible=True, value=debug_output), # Debug Box
                gr.update(visible=True), # Chat Area
                gr.update(visible=False), # Button
                gr.update(value=initial_history) # Chatbot with welcome message
            )
        except Exception as e:
            error_msg = f"Error initializing AI Assistant: {e}. Check your GOOGLE_API_KEY and model settings."
            return (
                gr.update(visible=True, value=error_msg), 
                gr.update(visible=False), 
                gr.update(visible=True), 
                gr.update(value=[])
            )
    else:
        error_message = "Please provide both a **Current Resume** and a **Job Description** to proceed."
        return (
            gr.update(visible=debug_mode, value=error_message), 
            gr.update(visible=False), 
            gr.update(visible=True), 
            gr.update(value=[])
        )

def chat_interface_fn(message: str, history: List[List[Optional[str]]], download_path_state: str) -> Tuple[List[List[Optional[str]]], str, gr.update]:
    """
    Gradio function to call the AI Assistant's chat method and handle file downloads.
    
    Returns: (Updated History, Updated Download Path State, Updated Download Link Component)
    """
    global assistant_instance

    if not assistant_instance:
        return history + [[message, "Error: The AI Assistant was not initialized. Please click 'Analyze and Start Chat' first."]], download_path_state, gr.update(visible=False)

    # 1. Call the AI Assistant to get the response and potential file path
    reply_text, file_path_from_tool = assistant_instance.chat(message, history)
    
    # Initialize updates
    new_download_path = download_path_state # Keep the existing path unless a new one is found
    download_link_update = gr.update(visible=False) # Start with hidden update

    if file_path_from_tool:
        # The tool successfully ran and returned a new file path
        new_download_path = file_path_from_tool
        
        # 2. Update the download link visibility and value
        download_link_update = gr.update(
            value=file_path_from_tool,
            visible=True
        )
        # Append a confirmation message to the chat
        reply_text += "\n\n**✅ The optimized resume is ready!** Please use the download link provided above the chat box."
    
    # 3. Append the user message and the assistant's reply to history
    history.append([message, reply_text])
    
    # 4. Return all updated outputs
    return history, new_download_path, download_link_update


# ----------------------------------------------------------------------
# --- GRADIO INTERFACE SETUP ---
# ----------------------------------------------------------------------

with gr.Blocks(title="Resume & Job Description Assistant") as app:
    gr.Markdown("# 📄 AI Career Match Assistant")
    
    # Hidden State components to hold the file paths and text for chat logic
    current_resume_text = gr.State(value="")
    job_description_text = gr.State(value="")
    resume_file_state = gr.State(value=None) 
    job_file_state = gr.State(value=None)
    download_file_path = gr.State(value=None) # State to hold the generated file path


    # --- Input Section ---
    with gr.Row():
        # Resume Input Column with Tabs
        with gr.Column():
            gr.Markdown("## 1. Current Resume")
            with gr.Tabs():
                with gr.TabItem("File Upload"):
                    resume_file_input = gr.File(
                        label="Upload Resume (PDF or Text File)", 
                        file_types=[".pdf", ".txt"],
                        type="filepath"
                    )
                    resume_file_input.change(fn=lambda x: x.name if x else None, inputs=resume_file_input, outputs=resume_file_state, queue=False)
                    resume_file_input.change(fn=lambda: "", inputs=None, outputs=current_resume_text, queue=False)
                
                with gr.TabItem("Paste Text"):
                    resume_text_input = gr.Textbox(
                        label="Paste Resume Text Here",
                        placeholder="Paste the text content of your resume.",
                        lines=10
                    )
                    resume_text_input.change(fn=lambda x: x, inputs=resume_text_input, outputs=current_resume_text, queue=False)
                    resume_text_input.change(fn=lambda: None, inputs=None, outputs=resume_file_state, queue=False)

        # Job Description Input Column with Tabs
        with gr.Column():
            gr.Markdown("## 2. Job Description")
            with gr.Tabs():
                with gr.TabItem("File Upload"):
                    job_file_input = gr.File(
                        label="Upload Job Description (PDF or Text File)", 
                        file_types=[".pdf", ".txt", ".doc", ".docx"],
                        type="filepath"
                    )
                    job_file_input.change(fn=lambda x: x.name if x else None, inputs=job_file_input, outputs=job_file_state, queue=False)
                    job_file_input.change(fn=lambda: "", inputs=None, outputs=job_description_text, queue=False)
                
                with gr.TabItem("Paste Text"):
                    job_text_input = gr.Textbox(
                        label="Paste Job Description Text Here",
                        placeholder="Paste the text content of the job description.",
                        lines=10
                    )
                    job_text_input.change(fn=lambda x: x, inputs=job_text_input, outputs=job_description_text, queue=False)
                    job_text_input.change(fn=lambda: None, inputs=None, outputs=job_file_state, queue=False)
                    
    # --- Control and Debug Section ---
    activate_button = gr.Button("Analyze and Start Chat", variant="primary")
    
    debug_output = gr.Textbox(
        label="Debug Output (File Path/Content Snippet)",
        visible=DEBUG_MODE,
        lines=10,
        interactive=False
    )
    
    # --- Chat Interface Section ---

    # File component that will become visible when the agent generates the resume
    download_link = gr.File(
        label="Download Optimized Resume (Available after analysis is complete)",
        visible=False, 
        type="filepath", 
        interactive=False # Must be set to False for download link functionality
    )

    chat_group = gr.Group(visible=False)
    with chat_group:
        gr.Markdown("## Career Match Coach")
        
        chatbot = gr.Chatbot(
            label="Career Assistant Chat",
            height=400,
            show_copy_button=True,
            show_label=True
        )
        
        chat_input = gr.Textbox(placeholder="Ask me how well your resume matches the job description...")
        
        # Connect the chat logic to the submission event
        chat_submit_event  = chat_input.submit(
            fn=chat_interface_fn, 
            inputs=[
                chat_input, 
                chatbot, 
                download_file_path # State holding the current path
            ], 
            outputs=[
                chatbot, 
                download_file_path, # Update the state with the new path
                download_link      # Update the download link component
            ]
        ).then(
            # Clear the chat input box after submission
            lambda: gr.update(value=""), 
            inputs=None, 
            outputs=chat_input, 
            queue=False
        )
        
        chat_submit_event.then(
            # We use a custom JavaScript function to trigger the download.
            # $0 is the first output component from the previous step (which is the chatbot here, but we can target the download_link state)
            # The more robust way is to specifically check the download_file_path state in JS.
            None, # No Python function needed here
            inputs=[download_link], # The Gradio component containing the path
            outputs=None,
            js="""
            (fileComponent) => {
                // fileComponent is the JSON object for the gr.File component
                if (fileComponent && fileComponent.value && fileComponent.value.name) {
                    // Create an anchor tag
                    const a = document.createElement('a');
                    // Gradio paths are usually prefixed with /file=
                    const downloadUrl = '/file=' + fileComponent.value.name;
                    
                    a.href = downloadUrl;
                    // Set the filename
                    a.download = fileComponent.value.orig_name || 'your_new_resume.html'; 
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    console.log('Auto-download triggered for:', fileComponent.value.name);
                }
            }
            """
        ).then(
            # Clear the chat input box after submission
            lambda: gr.update(value=""), 
            inputs=None, 
            outputs=chat_input, 
            queue=False
        )

    # --- Event Handling: The main logic connector ---
    activate_button.click(
        fn=activate_chat,
        inputs=[
            resume_file_state, 
            job_file_state,    
            current_resume_text,
            job_description_text,
            gr.State(value=DEBUG_MODE)
        ],
        outputs=[debug_output, chat_group, activate_button, chatbot]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()
import gradio as gr
import os 
import json
from typing import Tuple, List, Optional
from pypdf import PdfReader 
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from weasyprint import HTML

# --- Configuration ---
 
# Global state to hold the instance of the AI Assistant
agent = None
 
load_dotenv(override=True)
# --- Tool Functions (Agent Backend) ---

@function_tool
def resume_wrapper(final_summary: str, resume_detail: str, job_detail: str) -> dict:
    """
    Use this tool to generate the final optimized resume as an HTML file. 
    Call this only once you have enough information and are ready to conclude the conversation.
    
    :param final_summary: A concise summary (max 300 words) of the candidate's profile, including refinements made during the chat, to be used for the resume generation.
    :param resume_detail: The full content of the user's original resume.
    :param job_detail: The full content of the job description.
    """
    
    def resume_writing(final_summary: str, resume_detail: str, job_detail: str) -> str:
        """Generates the optimized resume content in JSON format."""
        writer_prompt = (
            "You are an expert resume writer. You will be given three inputs: "
            "1) the candidate's **original** resume, 2) the job description, and 3) the **final summary** based on the coaching session. "
            "Your task is to generate an optimized resume in **JSON format only** (no markdown, no code blocks). "
            "The JSON must incorporate the feedback from the final summary and align with the job description.\n\n"
            f"Candidate Summary:\n{final_summary}\n\n" 
            f"Original Resume:\n{resume_detail}\n\n"
            f"Job Description:\n{job_detail}\n\n"
        )
        # This function will be executed by an agent that has an LLM client.
        # We are assuming the agent will run this.
        # In the original code, it was using self.coach_ai.chat.completions.create
        # The new agent will handle this. For now, we'll just return a placeholder.
        # A proper implementation would involve another agent call here.
        # To keep it simple, we will simulate the call.
        from openai import OpenAI
        client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.getenv('GOOGLE_API_KEY'))
        response = client.chat.completions.create(model="gemini-2.5-flash-preview-05-20", messages=[{"role": "user", "content": writer_prompt}]).choices[0].message.content
        return response

    def resume_design(resume_json: str) -> str:
        """Converts the resume JSON into a clean, professional HTML layout."""
        design_prompt = (
            "You are an expert in resume design. You will receive a resume in JSON format, "
            "and your task is to convert it into a clean, professional HTML layout. "
            "The HTML should be properly structured and styled for later conversion into an A4 PDF document. "
            "Do not include markdown formatting or code blocks—return raw HTML only.\n\n"
            f"Resume JSON:\n{resume_json}\n\n"
        )
        from openai import OpenAI
        client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.getenv('GOOGLE_API_KEY'))
        response = client.chat.completions.create(model="gemini-2.5-flash-preview-05-20", messages=[{"role": "user", "content": design_prompt}]).choices[0].message.content
        return response

    print("Resume_wrapper: Starting generation and design...")
    
    # 1. Generate JSON resume using the final summary and original docs
    resume_answer = resume_writing(final_summary, resume_detail, job_detail) 
    
    # 2. Convert JSON to HTML design
    print("Resume_wrapper: Resume content generated. Starting HTML design...")
    design_answer = resume_design(resume_answer)
    
    # 3. Save the file
    output_filepath = "your_new_resume.html"
    with open(output_filepath, "w", encoding="utf-8") as file:
        file.write(design_answer)
        
    print("Resume_wrapper: Resume content generated. Converting to PDF ...")
    output_filepath = "your_new_resume.pdf"
    HTML(string=design_answer).write_pdf(output_filepath)
        
    print(f"Resume saved to {output_filepath}")

    # Return the file path for Gradio to handle the download
    return {"recorded": "ok", "file_path": output_filepath}


# ----------------------------------------------------------------------
# --- GRADIO INTERFACE FUNCTIONS ---
# ----------------------------------------------------------------------

def get_file_content(file_path: Optional[str], text_input: str) -> Tuple[str, str, Optional[str]]:
    """Reads content from file (including PDF extraction) or text input."""
    content = ""
    debug_msg = ""
    source_path = None

    if text_input:
        content = text_input
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
                
            except Exception as e:
                content = f"Error reading file {os.path.basename(source_path)}: {e}"
        else:
            content = f"Unsupported file type for {os.path.basename(source_path)}."

    return content, debug_msg, source_path

def activate_chat(
    resume_file: Optional[str], 
    job_file: Optional[str], 
    resume_text: str, 
    job_text: str
) -> Tuple[gr.update, gr.update, gr.update]:
    """Initializes the AI Assistant and reveals the chat interface."""
    global agent

    resume_content, _, _ = get_file_content(resume_file, resume_text)
    job_content, _, _ = get_file_content(job_file, job_text)

    is_resume_ready = bool(resume_content) and "Error reading file" not in resume_content
    is_job_ready = bool(job_content) and "Error reading file" not in job_content

    if is_resume_ready and is_job_ready:
        try:
            # 1. Define the Coach Prompt
            coach_prompt = (
                "You are a supportive and experienced career coach who specializes in tailoring resumes to match job descriptions effectively. "
                "Your role is to guide the candidate in creating a compelling and confident resume summary that aligns closely with the job they are applying for.\n\n"
                "You will be provided with the candidate's current resume and the job description. "
                "After reviewing them, you may ask the candidate a few short, focused questions to gather any missing or clarifying details. "
                "Once you have enough information, conclude the conversation by thanking the candidate. "
                "After that, call the 'resume_wrapper' tool to generate the optimized resume.\n\n"
                f"Current Resume (Snippet):\n{resume_content[:500]}...\n\n"
                f"Job Description (Snippet):\n{job_content[:500]}...\n\n"
                # Pass the full content to the tool through arguments
                f"When you call 'resume_wrapper', you MUST pass the full resume and job description content like this: "
                f"resume_wrapper(final_summary='...', resume_detail='''{resume_content}''', job_detail='''{job_content}''')"
            )
            
            # 2. Initialize the AI Agent
            gemini_client = AsyncOpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai", api_key=os.getenv('GOOGLE_API_KEY'))
            gemini_model = OpenAIChatCompletionsModel(model="gemini-2.5-flash-preview-05-20", openai_client=gemini_client)

            agent = Agent(
                name="CareerCoach",
                instructions=coach_prompt,
                tools=[resume_wrapper],
                model=gemini_model
            )
            
            # 3. Get initial message (optional, but good for user experience)
            # This part is simplified as the agent will respond to the user's first message.
            # We can create a dummy initial history.
            initial_history = [
                {"role": "assistant", "content": "Hello! I'm your career coach. To get started, please tell me a bit about your goals or ask me how well your resume matches the job description."}
            ]

            return (
                gr.update(visible=True), # Chat Area
                gr.update(visible=False), # Button
                gr.update(value=initial_history) # Chatbot with welcome message
            )
        except Exception as e:
            error_msg = f"Error initializing AI Agent: {e}. Check your OPENAI_API_KEY and model settings."
            return (
                gr.update(visible=True, value=error_msg), 
                gr.update(visible=False), 
                gr.update(visible=True), 
                gr.update(value=[])
            )
    else:
        error_message = "Please provide both a **Current Resume** and a **Job Description** to proceed."
        return (
            gr.update(visible=False, value=error_message), 
            gr.update(visible=False), 
            gr.update(visible=True), 
            gr.update(value=[])
        )

async def chat_interface_fn(message: str, history: List[dict[str, str]], download_path_state: str) -> Tuple[List[dict[str, str]], str, gr.update]:
    """
    Gradio function to call the AI Assistant's chat method and handle file downloads.
    
    Returns: (Updated History, Updated Download Path State, Updated Download Link Component)
    """
    global agent

    if not agent:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Error: The AI Agent was not initialized. Please click 'Analyze and Start Chat' first."})
        return history, download_path_state, gr.update(visible=False)

    # 1. Add user message to history
    history.append({"role": "user", "content": message})

    # 2. Call the AI Agent to get the response, passing a sanitized message list
    messages_for_agent = [{"role": item["role"], "content": item["content"]} for item in history]
    result = await Runner.run(agent, messages_for_agent)
    reply_text = result.final_output
    
    # Add assistant's response to history
    history.append({"role": "assistant", "content": reply_text})
    
    # Initialize updates
    new_download_path = download_path_state # Keep the existing path unless a new one is found
    download_link_update = gr.update(visible=False) # Start with hidden update

    # The tool output is not directly available in the same way.
    # We need to check if the tool was called and parse the output.
    # For simplicity, we'll check if the reply contains a hint that the file is ready.
    # A more robust solution would use sessions and inspect the tool outputs.
    if "resume is ready" in reply_text.lower() or "your_new_resume.pdf" in reply_text or "your_new_resume.html" in reply_text:
        # A bit of a hack: assume the file is ready if the agent says so.
        # file_path_from_tool = "your_new_resume.html"
        file_path_from_tool = "your_new_resume.pdf"
        if os.path.exists(file_path_from_tool):
            new_download_path = file_path_from_tool
            
            download_link_update = gr.update(
                value=file_path_from_tool,
                visible=True
            )
            # We don't need to add the confirmation to the reply_text, 
            # as it's already in the history.
            # reply_text += "\n\n**✅ The optimized resume is ready!** Please use the download link provided above the chat box."

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
            show_label=True,
            type="messages"
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
            job_description_text
        ],
        outputs=[chat_group, activate_button, chatbot]
    )

# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0", 
        server_port=int(os.environ.get('PORT', 7860))
    )
# ResumeCast: AI Career Match Assistant

ResumeCast is an intelligent web application that helps job seekers tailor their resumes to specific job descriptions using AI-powered coaching. The application features an interactive AI career coach that guides users through optimizing their resume content and automatically generates a professionally designed resume.

## 🚀 Features

- **Interactive AI Career Coach**: Chat with an AI assistant powered by Google's Gemini model to refine your resume
- **Flexible Input Options**: Upload files (PDF, TXT) or paste text directly for both resume and job description
- **Intelligent Resume Generation**: AI analyzes your resume against the job description and creates an optimized version
- **Professional HTML Output**: Generated resumes are formatted in clean, professional HTML ready for PDF conversion
- **Automatic Download**: Optimized resume is automatically downloaded once generation is complete
- **Real-time Chat Interface**: Gradio-powered web interface for seamless user interaction

## 🛠️ Technology Stack

- **Frontend**: Gradio web interface
- **AI Framework**: OpenAI Agents SDK
- **Language Model**: Google Gemini 2.5 Flash Preview
- **File Processing**: PyPDF for PDF text extraction
- **Environment Management**: python-dotenv

## 📋 Prerequisites

- Python 3.8 or higher
- Google API key for Gemini access
- Required Python packages (see requirements.txt)

## 🔧 Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repository-url>
   cd ResumeCast
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 🚀 Usage

1. **Start the application:**

   ```bash
   python main.py
   ```

2. **Open your browser** and navigate to the provided local URL (typically `http://127.0.0.1:7860`)

3. **Upload or paste your content:**

   - Add your current resume (PDF or text)
   - Add the target job description (PDF or text)

4. **Start the coaching session:**

   - Click "Analyze and Start Chat"
   - Interact with the AI career coach
   - Answer questions to help tailor your resume

5. **Receive your optimized resume:**
   - The AI will automatically generate and download your new resume
   - The file will be saved as `your_new_resume.html`

## 🏗️ Project Structure

```
ResumeCast/
├── main.py              # Main application file
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
├── .env                # Environment variables (create this)
└── your_new_resume.html # Generated resume output
```

## 🔍 How It Works

1. **Agent Initialization**: The application creates a "CareerCoach" agent using the OpenAI Agents SDK
2. **Content Analysis**: The AI analyzes your resume and job description to understand requirements
3. **Interactive Coaching**: The agent asks clarifying questions to gather additional context
4. **Resume Generation**:
   - Uses a specialized `resume_wrapper` tool
   - Generates optimized content in JSON format
   - Converts JSON to professional HTML layout
5. **File Output**: Saves the final resume as an HTML file ready for use

## 🛡️ Environment Variables

| Variable         | Description                           | Required |
| ---------------- | ------------------------------------- | -------- |
| `GOOGLE_API_KEY` | Your Google API key for Gemini access | Yes      |

## 📝 Dependencies

Key dependencies include:

- `gradio` - Web interface framework
- `openai-agents` - Agent SDK for AI interactions
- `pypdf` - PDF text extraction
- `python-dotenv` - Environment variable management
- `openai` - OpenAI client library

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

**Common Issues:**

1. **API Key Error**: Ensure your `GOOGLE_API_KEY` is correctly set in the `.env` file
2. **Module Import Error**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
3. **PDF Reading Issues**: Ensure uploaded PDFs contain readable text (not just images)
4. **Chat Interface Not Appearing**: Verify both resume and job description are provided before clicking "Analyze and Start Chat"

## 📞 Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.

---

**Made with ❤️ using AI and modern web technologies**

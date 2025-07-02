import google.generativeai as genai
import gradio as gr
import pandas as pd
from uuid import uuid4
import os

# --- SETUP YOUR GEMINI API KEY HERE ---
# Make sure to replace "YOUR_GEMINI_API_KEY" with your actual key.
try:
    # It's better practice to use environment variables for API keys.
    # For example: genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    genai.configure(api_key="AIzaSyDHBIXCY7b0rPNCtTEUj4JhylQ14wO0ozA")
except Exception as e:
    print(f"API Key Configuration Error: {e}\nPlease set your Gemini API key.")

# --- MODEL INITIALIZATION ---
# Initialize the Gemini model
try:
    gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
    chat = gemini_model.start_chat()
    print("Gemini model initialized successfully.")
except Exception as e:
    print(f"Model Initialization Error: {e}")
    gemini_model = None

# --- DATA STORAGE ---
# In-memory storage for the session data. A new DataFrame is created for each session.
session_data = []

# --- CORE FUNCTIONS ---

def generate_poem(prompt, session_type):
    """
    Generates an Arabic poem using the Gemini model based on the user's prompt.
    The prompt to the AI is enhanced to guide it better.
    """
    if not gemini_model:
        return "Error: Gemini model is not initialized. Please check your API key.", ""

    # We can craft a more specific instruction for the model
    enhanced_prompt = f"""
    Based on the following theme, please generate a culturally rich and authentic Arabic poem.
    The user providing the theme is a '{session_type}'.
    Theme: "{prompt}"
    """
    try:
        response = chat.send_message(enhanced_prompt)
        poem_text = response.text.strip()
        return poem_text
    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

def save_record(session_type, prompt, generated_poem,
                meter_score, meter_comment,
                harmony_score, harmony_comment,
                imagery_score, imagery_comment,
                depth_score, depth_comment,
                culture_score, culture_comment,
                flow_score, flow_comment,
                originality_score, originality_comment,
                usefulness_score, usefulness_comment,
                decision):
    """
    Saves all the input and evaluation data for one record into the session_data list
    and returns an updated DataFrame for display.
    """
    record = {
        "Session ID": uuid4().hex[:8],
        "Session Type": session_type,
        "User Prompt": prompt,
        "Generated Poem": generated_poem,
        "Meter Score (1-5)": meter_score,
        "Meter Comment": meter_comment,
        "Sound Harmony Score (1-5)": harmony_score,
        "Sound Harmony Comment": harmony_comment,
        "Imagery Score (1-5)": imagery_score,
        "Imagery Comment": imagery_comment,
        "Emotional Depth Score (1-5)": depth_score,
        "Emotional Depth Comment": depth_comment,
        "Cultural Relevance Score (1-5)": culture_score,
        "Cultural Relevance Comment": culture_comment,
        "Coherence/Flow Score (1-5)": flow_score,
        "Coherence/Flow Comment": flow_comment,
        "Originality Score (1-5)": originality_score,
        "Originality Comment": originality_comment,
        "Usefulness Score (1-5)": usefulness_score,
        "Usefulness Comment": usefulness_comment,
        "Final Decision": decision
    }
    session_data.append(record)
    
    # Create a pandas DataFrame to display the collected data
    df = pd.DataFrame(session_data)
    return df, "Record saved successfully!"

def download_data():
    """
    Saves the session data to a CSV file and returns the file path for download.
    """
    if not session_data:
        return None # Return None if there's no data to save
        
    df = pd.DataFrame(session_data)
    # The file is saved in the same directory where the script is run
    file_path = f"arabic_poetry_evaluation_{uuid4().hex[:6]}.csv"
    df.to_csv(file_path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
    return file_path

# --- GRADIO WEB APPLICATION INTERFACE ---

with gr.Blocks(theme=gr.themes.Soft(), title="AI Arabic Poet's Assistant") as app:
    gr.Markdown("# AI Arabic Poet's Assistant")
    gr.Markdown("A research tool to generate and evaluate AI-created Arabic poetry.")

    with gr.Row():
        # Inputs Column
        with gr.Column(scale=1):
            gr.Markdown("## 1. Input")
            session_type = gr.Radio(choices=["Novice", "Expert"], label="Select Your Expertise Level", value="Novice")
            user_prompt = gr.Textbox(lines=5, label="Enter Poem Description (in Arabic)", placeholder="اكتب وصفاً أو فكرة شعرية هنا...")
            
            with gr.Row():
                generate_btn = gr.Button("Generate Poem", variant="primary")
                regenerate_btn = gr.Button("Regenerate")

        # Outputs Column
        with gr.Column(scale=2):
            gr.Markdown("## 2. Generated Poem")
            generated_poem_output = gr.Textbox(lines=10, label="AI-Generated Poem", interactive=False)

    gr.Markdown("---")
    gr.Markdown("## 3. Evaluation Form")

    # Using a list to create the form dynamically
    criteria = [
        "Meter and Rhythm (الوزن والإيقاع)",
        "Sound Harmony (القافية والانسجام الصوتي)",
        "Imagery and Metaphor (الصورة الشعرية)",
        "Emotional Depth (العمق العاطفي)",
        "Cultural Relevance (الأصالة الثقافية)",
        "Coherence and Flow (التماسك والسلاسة)",
        "Originality (الأصالة والابتكار)",
        "Usefulness to Poet (الفائدة للشاعر)"
    ]
    
    evaluation_inputs = []
    for criterion in criteria:
        with gr.Row():
            score = gr.Dropdown(choices=[1, 2, 3, 4, 5], label=f"{criterion} Score", scale=1)
            comment = gr.Textbox(label=f"{criterion} Comment", scale=3)
            evaluation_inputs.extend([score, comment])

    gr.Markdown("---")
    gr.Markdown("## 4. Final Decision & Save")
    
    with gr.Row():
        final_decision = gr.Radio(choices=["Approve", "Modify", "Reject"], label="Final Decision", value="Approve")
        save_btn = gr.Button("💾 Save Record", variant="primary")
        status_message = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        download_btn = gr.Button("Download Session Data (CSV)")
        download_file_output = gr.File(label="Download Link")

    gr.Markdown("## Session Data Log")
    session_table = gr.DataFrame(label="Saved Records")

    # --- CONNECTING FUNCTIONS TO UI COMPONENTS ---

    # Generate Button Action
    generate_btn.click(
        fn=generate_poem,
        inputs=[user_prompt, session_type],
        outputs=[generated_poem_output]
    )
    
    # Regenerate Button Action
    regenerate_btn.click(
        fn=generate_poem,
        inputs=[user_prompt, session_type],
        outputs=[generated_poem_output]
    )

    # Save Button Action
    save_btn.click(
        fn=save_record,
        inputs=[session_type, user_prompt, generated_poem_output] + evaluation_inputs + [final_decision],
        outputs=[session_table, status_message]
    )

    # Download Button Action
    download_btn.click(
        fn=download_data,
        outputs=[download_file_output]
    )

# Launch the application
app.launch(server_name="0.0.0.0", server_port=7860)

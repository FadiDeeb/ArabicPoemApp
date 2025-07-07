import google.generativeai as genai
import gradio as gr
import pandas as pd
from uuid import uuid4
import os
from scipy.stats import ttest_ind # <-- NEW: For statistical t-tests
import matplotlib.pyplot as plt # <-- NEW: For plotting

# --- SETUP AND MODEL INITIALIZATION (No Changes) ---
try:
    genai.configure(api_key="AIzaSyDHBIXCY7b0rPNCtTEUj4JhylQ14wO0ozA") # <-- IMPORTANT: SET YOUR KEY
except Exception as e:
    print(f"API Key Configuration Error: {e}\nPlease set your Gemini API key.")

try:
    gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
    chat = gemini_model.start_chat()
    print("Gemini model initialized successfully.")
except Exception as e:
    print(f"Model Initialization Error: {e}")
    gemini_model = None

# --- DATA STORAGE (No Changes) ---
session_data = []

# --- CORE POEM GENERATION FUNCTIONS (No Changes) ---

def generate_poem(prompt, session_type):
    if not gemini_model:
        return "Error: Gemini model is not initialized. Please check your API key."
    enhanced_prompt = f"""
    Based on the following theme, please generate a culturally rich and authentic Arabic poem.
    The user providing the theme is a '{session_type}'.
    Theme: "{prompt}"
    """
    try:
        response = chat.send_message(enhanced_prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

def save_record(session_type, prompt, generated_poem,
                meter_score, meter_comment, harmony_score, harmony_comment,
                imagery_score, imagery_comment, depth_score, depth_comment,
                culture_score, culture_comment, flow_score, flow_comment,
                originality_score, originality_comment, usefulness_score, usefulness_comment,
                decision):
    record = {
        "Session ID": uuid4().hex[:8], "Session Type": session_type, "User Prompt": prompt,
        "Generated Poem": generated_poem, "Meter Score (1-5)": meter_score, "Meter Comment": meter_comment,
        "Sound Harmony Score (1-5)": harmony_score, "Sound Harmony Comment": harmony_comment,
        "Imagery Score (1-5)": imagery_score, "Imagery Comment": imagery_comment,
        "Emotional Depth Score (1-5)": depth_score, "Emotional Depth Comment": depth_comment,
        "Cultural Relevance Score (1-5)": culture_score, "Cultural Relevance Comment": culture_comment,
        "Coherence/Flow Score (1-5)": flow_score, "Coherence/Flow Comment": flow_comment,
        "Originality Score (1-5)": originality_score, "Originality Comment": originality_comment,
        "Usefulness Score (1-5)": usefulness_score, "Usefulness Comment": usefulness_comment,
        "Final Decision": decision
    }
    session_data.append(record)
    df = pd.DataFrame(session_data)
    return df, "Record saved successfully!"

def download_data():
    if not session_data: return None
    df = pd.DataFrame(session_data)
    file_path = f"arabic_poetry_evaluation_{uuid4().hex[:6]}.csv"
    df.to_csv(file_path, index=False, encoding='utf-8-sig')
    return file_path

# --- NEW: STATISTICAL ANALYSIS FUNCTION ---

SCORE_COLUMNS = [
    "Meter and Rhythm (الوزن والإيقاع) Score (1-5)",
    "Sound Harmony (القافية والانسجام الصوتي) Score (1-5)",
    "Imagery and Metaphor (الصورة الشعرية) Score (1-5)",
    "Emotional Depth (العمق العاطفي) Score (1-5)",
    "Cultural Relevance (الأصالة الثقافية) Score (1-5)",
    "Coherence and Flow (التماسك والسلاسة) Score (1-5)",
    "Originality (الأصالة والابتكار) Score (1-5)",
    "Usefulness to Poet (الفائدة للشاعر) Score (1-5)"
]

def perform_analysis(novice_file, expert_file):
    """
    Performs t-tests and generates a boxplot from uploaded novice and expert CSV files.
    """
    if novice_file is None or expert_file is None:
        return None, None, "Please upload both novice and expert data files."

    try:
        novice_df = pd.read_csv(novice_file.name)
        expert_df = pd.read_csv(expert_file.name)
    except Exception as e:
        return None, None, f"Error reading files: {e}"

    # 1. Perform T-Tests
    results = []
    for col in SCORE_COLUMNS:
        if col not in novice_df.columns or col not in expert_df.columns:
            continue # Skip if a column is missing
        
        # Drop missing values for the specific test
        novice_scores = novice_df[col].dropna()
        expert_scores = expert_df[col].dropna()

        if len(novice_scores) < 2 or len(expert_scores) < 2:
            continue # Not enough data to perform a t-test

        t_stat, p_value = ttest_ind(novice_scores, expert_scores, equal_var=False) # Welch's t-test is safer
        
        significance = "✅ Significant" if p_value < 0.05 else "❌ Not Significant"
        criterion_name = col.replace(' Score (1-5)', '').replace(' (', '\n(') # For better display
        
        results.append({
            "Criterion": criterion_name,
            "p-value": f"{p_value:.6f}",
            "Interpretation": significance
        })
    
    results_df = pd.DataFrame(results)

    # 2. Create Boxplot Visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Prepare data for plotting
    plot_data = []
    labels = []
    for col in SCORE_COLUMNS:
        plot_data.append(novice_df[col].dropna())
        plot_data.append(expert_df[col].dropna())
        labels.append(f"{col.replace(' Score (1-5)', '')}\n(Novice)")
        labels.append(f"{col.replace(' Score (1-5)', '')}\n(Expert)")

    # Create the boxplot
    bp = ax.boxplot(plot_data, patch_artist=True, labels=labels)
    
    # Color the boxes for clarity
    for i, box in enumerate(bp['boxes']):
        if i % 2 == 0: # Novice
            box.set_facecolor('lightblue')
        else: # Expert
            box.set_facecolor('lightgreen')

    ax.set_title('Comparison of Scores: Novice vs. Expert Sessions', fontsize=16)
    ax.set_ylabel('Scores (1-5)', fontsize=12)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.yaxis.grid(True)
    plt.tight_layout() # Adjust layout to make room for labels

    # Save plot to a temporary file to display in Gradio
    plot_path = "analysis_boxplot.png"
    plt.savefig(plot_path)
    plt.close(fig) # Close the figure to free memory

    return results_df, plot_path, "Analysis complete."


# --- GRADIO WEB APPLICATION INTERFACE (Now with Tabs) ---

with gr.Blocks(theme=gr.themes.Soft(), title="AI Arabic Poet's Assistant") as app:
    gr.Markdown("# AI Arabic Poet's Assistant")
    gr.Markdown("A research tool to generate, evaluate, and analyze AI-created Arabic poetry.")

    with gr.Tabs():
        # --- TAB 1: POEM GENERATION AND EVALUATION ---
        with gr.TabItem("Poem Generation & Evaluation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 1. Input")
                    session_type = gr.Radio(choices=["Novice", "Expert"], label="Select Your Expertise Level", value="Novice")
                    user_prompt = gr.Textbox(lines=5, label="Enter Poem Description (in Arabic)", placeholder="اكتب وصفاً أو فكرة شعرية هنا...")
                    with gr.Row():
                        generate_btn = gr.Button("Generate Poem", variant="primary")
                        regenerate_btn = gr.Button("Regenerate")
                with gr.Column(scale=2):
                    gr.Markdown("## 2. Generated Poem")
                    generated_poem_output = gr.Textbox(lines=10, label="AI-Generated Poem", interactive=False)
            
            gr.Markdown("---")
            gr.Markdown("## 3. Evaluation Form")
            evaluation_inputs = []
            for criterion in SCORE_COLUMNS:
                with gr.Row():
                    clean_criterion_name = criterion.replace(' Score (1-5)', '')
                    score = gr.Dropdown(choices=[1, 2, 3, 4, 5], label=f"{clean_criterion_name} Score", scale=1)
                    comment = gr.Textbox(label=f"{clean_criterion_name} Comment", scale=3)
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

        # --- NEW TAB 2: STATISTICAL ANALYSIS ---
        with gr.TabItem("Statistical Analysis"):
            gr.Markdown("## T-Test and Visualization")
            gr.Markdown("Upload the CSV files generated from the 'Poem Generation' tab to compare the results.")
            
            with gr.Row():
                novice_file_input = gr.File(label="Upload Novice Data (CSV)")
                expert_file_input = gr.File(label="Upload Expert Data (CSV)")
            
            analyze_btn = gr.Button("Run Analysis", variant="primary")
            analysis_status = gr.Textbox(label="Analysis Status", interactive=False)
            
            gr.Markdown("### T-Test Results")
            analysis_results_table = gr.DataFrame(label="P-value Interpretation")
            
            gr.Markdown("### Boxplot Visualization")
            analysis_plot_output = gr.Image(label="Score Distribution: Novice vs. Expert")

    # --- CONNECTING FUNCTIONS TO UI COMPONENTS ---

    # Tab 1 Connections
    generate_btn.click(fn=generate_poem, inputs=[user_prompt, session_type], outputs=[generated_poem_output])
    regenerate_btn.click(fn=generate_poem, inputs=[user_prompt, session_type], outputs=[generated_poem_output])
    save_btn.click(
        fn=save_record,
        inputs=[session_type, user_prompt, generated_poem_output] + evaluation_inputs + [final_decision],
        outputs=[session_table, status_message]
    )
    download_btn.click(fn=download_data, outputs=[download_file_output])

    # Tab 2 Connections
    analyze_btn.click(
        fn=perform_analysis,
        inputs=[novice_file_input, expert_file_input],
        outputs=[analysis_results_table, analysis_plot_output, analysis_status]
    )

# Launch the application
app.launch(server_name="0.0.0.0", server_port=7860)

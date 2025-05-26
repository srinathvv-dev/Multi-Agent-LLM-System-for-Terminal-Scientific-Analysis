# ## @file
# #  @brief Perform automated technical analysis on plot images using OCR and LLM, and generate a PDF report.
# # To run this  program with maximum GPU effieciency: python3 analyzer.py my_plot.png --model mistral-7b-instruct-v0.1.Q4_K_M.gguf --output output.pdf --gpu_layers 80


# import os  # Provides functions to interact with the operating system
# import argparse  # For parsing command-line arguments
# from PIL import Image  # Python Imaging Library for opening and processing images
# from fpdf import FPDF  # Library to create PDF files
# from llama_cpp import Llama  # Used to load and run LLaMA LLM models
# import pytesseract  # OCR (optical character recognition) library
# from datetime import datetime  # To get current date and time
# import multiprocessing  # Allows using multiple CPU cores


# ## @class PlotAnalyzer
# #  @brief Class that loads an LLM, extracts text from plot images, analyzes the content, and generates a report.
# class PlotAnalyzer:
#     ## @brief Initializes the PlotAnalyzer with model settings and loads the model.
#     #  @param model_path Path to the LLM model file.
#     #  @param gpu_layers Number of layers to offload to the GPU.
#     def __init__(self, model_path, gpu_layers):
#         self.model_path = model_path  # Store the model path
#         self.gpu_layers = gpu_layers  # Store number of GPU layers to use
#         self.llm = None  # Placeholder for the LLM instance
#         self.load_model()  # Load the model immediately on initialization

#     ## @brief Loads the LLM model with specified GPU configuration.
#     def load_model(self):
#         print("Loading model with GPU acceleration...")  # Inform user of model loading
#         self.llm = Llama(
#             model_path=self.model_path,  # Path to model file
#             n_ctx=8192,  # Set context window size
#             n_threads=multiprocessing.cpu_count(),  # Use all CPU cores available
#             n_gpu_layers=self.gpu_layers,  # Number of layers offloaded to GPU
#             main_gpu=0,  # Use GPU 0
#             n_batch=512,  # Batch size for inference
#             seed=42,  # Set seed for reproducibility
#             verbose=True  # Enable detailed logging
#         )
#         print("Model loaded successfully")  # Confirmation message

#     ## @brief Extracts readable text from an image using OCR.
#     #  @param image_path File path to the image.
#     #  @return Extracted text from the image as a string.
#     def extract_text_from_image(self, image_path):
#         try:
#             img = Image.open(image_path)  # Open the image file
#             img = img.convert('L')  # Convert to grayscale
#             img = img.point(lambda x: 0 if x < 128 else 255)  # Binarize image (black & white)
#             config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789().,-:/'  # OCR configuration
#             text = pytesseract.image_to_string(img, config=config)  # Extract text using Tesseract
#             return text.strip()  # Return text with no leading/trailing whitespace
#         except Exception as e:
#             print(f"OCR error: {e}")  # Print error if OCR fails
#             return ""  # Return empty string on failure

#     ## @brief Generates a detailed analysis report from an image using the LLM.
#     #  @param image_path Path to the image for analysis.
#     #  @return A markdown-formatted analysis string.
#     def generate_analysis(self, image_path):
#         extracted_text = self.extract_text_from_image(image_path)  # Extract text from image
#         prompt = f"""  # Formulate the prompt for the LLM
# STRICT INSTRUCTIONS: You MUST format your response EXACTLY as follows. Only include directly observable data. 

# [Extracted Text from Image]  
# {extracted_text}

# 1. GRAPH OVERVIEW
# - Chart type: 
# - Purpose: 
# - Key visual elements:
# - Axes: 
#   - X: [label], [unit], [scale], [range]
#   - Y: [label], [unit], [scale], [range]
# - Legends/annotations: 

# 2. DATA SUMMARY TABLE
# | Data Series | Min Value | Max Value | Average | Key Trends | Anomalies |
# |-------------|-----------|-----------|---------|------------|-----------|
# | [Name]      | [Value]   | [Value]   |         | [Pattern]  |           |

# 3. QUANTITATIVE METRICS  
# - Total visible data points (approximate if needed)  
# - Peak values with coordinates (e.g., X=10, Y=82)  
# - Rate of change (linear slope, % change where visible)  
# - Variability (range or standard deviation if inferable from plot)

# 4. TECHNICAL INTERPRETATION  
# - What the data likely represents scientifically  
# - Logical causes for visible patterns or fluctuations  
# - Link to real-world applications or systems

# 5. LIMITATIONS & UNCERTAINTIES  
# - Issues affecting clarity (e.g., poor resolution, missing axis labels)  
# - Incomplete data ranges or unknown parameters  
# - Visual artifacts or design elements that might mislead

# 6. RECOMMENDATIONS  
# - Additional analyses to clarify insights  
# - Visualization improvements (e.g., color, labeling, axis scaling)  
# - Suggested next steps (e.g., collect more data, replot with error bars)

# Use a formal technical style. Structure your response using bullet points, markdown tables, and short, metric-focused observations. Do not invent or fabricate any data not visible in the image.

#         """
#         response = self.llm.create_chat_completion(  # Call the model to generate analysis
#             messages=[{"role": "user", "content": prompt}],  # Single-user message
#             max_tokens=4096,  # Set maximum response length
#             temperature=0.3,  # Control creativity
#             top_p=0.9,  # Nucleus sampling value
#             repeat_penalty=1.1  # Discourage repetition
#         )
#         return response['choices'][0]['message']['content']  # Return the generated content

#     ## @brief Creates and saves a formatted PDF report from the analysis text.
#     #  @param analysis_text Text content of the analysis.
#     #  @param output_path Destination path for the PDF file.
#     def create_pdf_report(self, analysis_text, output_path):
#         pdf = FPDF()  # Initialize PDF object
#         pdf.set_auto_page_break(auto=True, margin=15)  # Set auto page breaks
#         pdf.add_page()  # Add first page

#         font_family = 'Arial'  # Set font to Arial

#         pdf.set_font(font_family, 'B', 16)  # Set bold title font
#         pdf.cell(0, 10, "Technical Plot Analysis Report", ln=True, align="C")  # Add title
#         pdf.ln(10)  # Add spacing

#         pdf.set_font(font_family, '', 10)  # Set font for metadata
#         pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)  # Add date
#         pdf.ln(10)

#         pdf.set_font(font_family, '', 12)  # Set main body font

#         for line in analysis_text.split('\n'):  # Process line by line
#             line = line.strip()  # Remove whitespace
#             if not line:
#                 pdf.ln(5)  # Add vertical space for empty lines
#                 continue

#             if line.startswith("###"):  # Subsection header
#                 pdf.set_font(font_family, 'B', 13)
#                 pdf.set_text_color(0, 51, 102)  # Use blue-ish color
#                 pdf.multi_cell(0, 10, line.replace("###", "").strip())
#                 pdf.set_font(font_family, '', 12)
#                 pdf.set_text_color(0, 0, 0)
#                 pdf.ln(2)
#             elif line.startswith("##"):  # Section header
#                 pdf.set_font(font_family, 'B', 14)
#                 pdf.multi_cell(0, 10, line.replace("##", "").strip())
#                 pdf.set_font(font_family, '', 12)
#                 pdf.ln(1)
#             elif line.startswith("- "):  # Bullet point
#                 bullet = '*'
#                 pdf.multi_cell(0, 8, f"{bullet} {line[2:]}")
#             else:
#                 try:
#                     pdf.multi_cell(0, 8, line)  # Normal text
#                 except:
#                     fallback = line.encode('ascii', 'replace').decode('ascii')  # Handle encoding issues
#                     pdf.multi_cell(0, 8, fallback)

#         try:
#             pdf.output(output_path)  # Try to save the PDF
#             print(f"PDF saved: {output_path}")
#         except Exception as e:
#             print(f"Error saving PDF: {e}")  # Print error if save fails
#             with open(output_path, 'wb') as f:  # Fallback save
#                 f.write(pdf.output(dest='S').encode('latin-1', 'replace'))
#             print("Fallback save successful.")

# ## @brief Entry point for the script. Parses arguments and runs the analysis pipeline.
# def main():
#     parser = argparse.ArgumentParser(description="High-Performance Plot Analysis with Mistral")  # Set up argument parser
#     parser.add_argument("image_path", help="Path to the plot image file")  # Required: image path
#     parser.add_argument("--model", default="mistral-7b-instruct-v0.1.Q4_K_M.gguf", help="Path to the GGUF model file")  # Optional: model path
#     parser.add_argument("--output", default="technical_plot_analysis.pdf", help="Output PDF file path")  # Optional: output PDF path
#     parser.add_argument("--gpu_layers", type=int, default=60, help="Number of GPU layers to offload")  # Optional: GPU layers

#     args = parser.parse_args()  # Parse all arguments

#     print(f"Using {args.gpu_layers} GPU layers on NVIDIA Quadro RTX 6000...")  # Print selected config
#     analyzer = PlotAnalyzer(args.model, args.gpu_layers)  # Initialize analyzer

#     print("Generating analysis...")
#     analysis = analyzer.generate_analysis(args.image_path)  # Generate report text

#     print("Creating PDF report...")
#     analyzer.create_pdf_report(analysis, args.output)  # Save report as PDF

#     print("✅ Analysis complete!")  # Done

# ## @brief Runs the main function when the script is executed directly.
# if __name__ == "__main__":
#     main()

import os
import argparse
from PIL import Image
from fpdf import FPDF
from llama_cpp import Llama
import pytesseract
from datetime import datetime
import multiprocessing

class PlotAnalyzer:
    """High-precision scientific plot analyzer with template enforcement."""
    
    def __init__(self, model_path, gpu_layers):
        self.model_path = model_path
        self.gpu_layers = gpu_layers
        self.llm = None
        self.load_model()

    def load_model(self):
        """Load the LLM with optimized GPU settings."""
        print(f"Loading model with {self.gpu_layers} GPU layers...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=8192,
            n_threads=multiprocessing.cpu_count(),
            n_gpu_layers=self.gpu_layers,
            main_gpu=0,
            n_batch=512,
            seed=42,
            verbose=True
        )
        print("Model ready")

    def extract_text_from_image(self, image_path):
        """Enhanced OCR with preprocessing."""
        try:
            img = Image.open(image_path)
            # Preprocessing pipeline
            img = img.convert('L')  # Grayscale
            img = img.point(lambda x: 0 if x < 128 else 255)  # Binarize
            config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(img, config=config)
            return text.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def generate_analysis(self, image_path):
        """Generate analysis with strict template enforcement."""
        extracted_text = self.extract_text_from_image(image_path)
        
        prompt = f"""
        STRICT ANALYSIS TEMPLATE - COMPLETE ALL SECTIONS:

        [Extracted Text from Image]  
        {extracted_text if extracted_text else "No text detected"}

        1. GRAPH OVERVIEW
        - Chart type: 
        - Purpose: 
        - Key visual elements:
        - Axes:
          - X-axis: [label], [unit], [scale type], [range]
          - Y-axis: [label], [unit], [scale type], [range]
        - Legends/annotations: 

        2. DATA SUMMARY TABLE
        | Data Series | Min Value | Max Value | Key Trends | Anomalies |
        |-------------|-----------|-----------|------------|-----------|
        |             |           |           |            |           |

        3. QUANTITATIVE METRICS
        - Total data points: 
        - Peak values: 
        - Rate of change: 
        - Variability: 

        4. TECHNICAL INTERPRETATION
        - Scientific representation: 
        - Pattern causes: 
        - Real-world applications: 

        5. LIMITATIONS
        - Clarity issues: 
        - Missing data: 
        - Potential artifacts: 

        6. RECOMMENDATIONS
        - Additional analyses: 
        - Visualization improvements: 
        - Next steps: 

        RULES:
        1. Only report directly observable data
        2. Use metric units where visible
        3. Mark unavailable data as "Not specified"
        4. Maintain this exact structure
        """

        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.1,  # Highly deterministic
            top_p=0.5,
            repeat_penalty=1.3  # Strong anti-repetition
        )
        
        return self._validate_response(response['choices'][0]['message']['content'])

    def _validate_response(self, text):
        """Ensure complete template compliance."""
        required = [
            "1. GRAPH OVERVIEW",
            "2. DATA SUMMARY TABLE",
            "3. QUANTITATIVE METRICS",
            "4. TECHNICAL INTERPRETATION",
            "5. LIMITATIONS",
            "6. RECOMMENDATIONS"
        ]
        
        for section in required:
            if section not in text:
                text += f"\n\n{section}\n- [REQUIRED SECTION MISSING]"
        
        return text

    def create_pdf_report(self, analysis_text, output_path):
        """Generate professionally formatted PDF."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Header
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, "SCIENTIFIC PLOT ANALYSIS REPORT", ln=True, align='C')
        pdf.ln(10)
        
        # Metadata
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(15)
        
        # Content formatting
        pdf.set_font('Helvetica', '', 11)
        current_indent = 0
        
        for line in analysis_text.split('\n'):
            line = line.strip()
            
            # Section headers
            if line.startswith(('1. ', '2. ', '3. ', '4. ', '5. ', '6. ')):
                pdf.set_font('Helvetica', 'B', 12)
                pdf.set_text_color(0, 0, 128)  # Navy blue
                pdf.cell(0, 8, line, ln=True)
                pdf.set_font('Helvetica', '', 11)
                pdf.set_text_color(0, 0, 0)
                current_indent = 0
            
            # Table handling
            elif '|' in line and '---' not in line:
                cols = [c.strip() for c in line.split('|') if c]
                if len(cols) >= 3:  # Minimum valid table row
                    col_width = 40
                    for i, col in enumerate(cols):
                        pdf.cell(col_width, 8, col[:30], border=1)
                    pdf.ln()
            
            # Bullet points
            elif line.startswith('- '):
                pdf.cell(10)  # Indent
                pdf.cell(0, 8, line[2:], ln=True)
            
            # Regular text
            elif line:
                pdf.multi_cell(0, 8, line)
            
            # Empty lines
            else:
                pdf.ln(5)
        
        # Footer
        pdf.set_y(-15)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.cell(0, 10, "Generated by PlotAnalyzer v2.0", align='C')
        
        try:
            pdf.output(output_path)
            print(f"Report saved to {output_path}")
        except Exception as e:
            print(f"PDF Error: {e}")
            # Fallback save
            with open(output_path.replace('.pdf','_FALLBACK.pdf'), 'wb') as f:
                f.write(pdf.output(dest='S').encode('latin-1'))

def main():
    parser = argparse.ArgumentParser(
        description="Automated Technical Plot Analysis System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("image_path", help="Input plot image file")
    parser.add_argument(
        "--model", 
        default="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--output", 
        default="analysis_report.pdf",
        help="Output PDF filename"
    )
    parser.add_argument(
        "--gpu_layers", 
        type=int, 
        default=80,
        help="Number of GPU layers to offload"
    )
    
    args = parser.parse_args()
    
    analyzer = PlotAnalyzer(args.model, args.gpu_layers)
    
    print("\n" + "="*50)
    print("Starting analysis pipeline...")
    analysis = analyzer.generate_analysis(args.image_path)
    
    print("\nAnalysis complete. Generating PDF...")
    analyzer.create_pdf_report(analysis, args.output)
    
    print("\n" + "="*50)
    print("✅ Report generation successful!")
    print(f"Output file: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
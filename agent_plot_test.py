import os
import torch
from PIL import Image
from transformers import (
    CLIPProcessor, 
    CLIPModel,
    DonutProcessor,
    VisionEncoderDecoderModel,
    BlipForConditionalGeneration,
    BlipProcessor
)
from llama_cpp import Llama
import textwrap
import re
from typing import Optional

class ScientificPlotAnalyzer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # CLIP model - keep in float32 for stability
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        
        # Donut model - keep in float32
        self.donut_processor = DonutProcessor.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-docvqa"
        )
        self.donut_model = VisionEncoderDecoderModel.from_pretrained(
            "naver-clova-ix/donut-base-finetuned-docvqa"
        ).to(self.device)
        
        # BLIP model - keep in float32
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(self.device)
        
        # LLM setup
        self.llm = Llama(
            model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            n_ctx=8192,
            n_gpu_layers=40,
            n_threads=16
        )

    def analyze_plot(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            
            # CLIP analysis
            clip_inputs = self.clip_processor(
                text=["scientific plot", "data visualization", "research chart"],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                clip_outputs = self.clip_model(**clip_inputs)
                clip_probs = clip_outputs.logits_per_image.softmax(dim=1)
            
            # Donut analysis
            donut_inputs = self.donut_processor(
                image, 
                return_tensors="pt"
            ).to(self.device)
                
            pixel_values = donut_inputs.pixel_values
            
            # Create decoder inputs safely
            decoder_input_ids = torch.tensor(
                [[self.donut_processor.tokenizer.pad_token_id]],
                dtype=torch.long,
                device=self.device
            )
            
            with torch.no_grad():
                donut_output = self.donut_model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=512,
                    early_stopping=True,
                    pad_token_id=self.donut_processor.tokenizer.pad_token_id,
                    eos_token_id=self.donut_processor.tokenizer.eos_token_id,
                    num_beams=3
                )
            
            donut_result = self.donut_processor.batch_decode(
                donut_output, skip_special_tokens=True
            )[0]
            
            # BLIP analysis
            blip_inputs = self.blip_processor(
                image, 
                "A detailed scientific analysis of this plot including: "
                "1. Axis labels and units 2. Data trends 3. Statistical annotations",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                blip_output = self.blip_model.generate(
                    **blip_inputs,
                    max_new_tokens=512,
                    num_beams=5
                )
            
            blip_result = self.blip_processor.decode(
                blip_output[0], skip_special_tokens=True
            )
            
            # LLM synthesis
            analysis_prompt = f"""
            SCIENTIFIC PLOT ANALYSIS TASK
            
            [VISUAL ANALYSIS RESULTS]
            {donut_result}
            
            [DETAILED CAPTION]
            {blip_result}
            
            Generate a comprehensive technical report with these sections:
            1. Plot Identification
            2. Axis Specifications
            3. Data Series Analysis
            4. Statistical Findings
            5. Scientific Interpretation
            6. Quality Assessment
            
            Use precise technical language and include all quantitative details.
            """
            
            llm_response = self.llm.create_completion(
                analysis_prompt,
                max_tokens=4096,
                temperature=0.1,
                top_p=0.9,
                repeat_penalty=1.1
            )
            
            return llm_response['choices'][0]['text']
            
        except Exception as e:
            return f"Analysis failed: {str(e)}"

if __name__ == "__main__":
    try:
        analyzer = ScientificPlotAnalyzer(device="cuda")
        
        images = [f for f in os.listdir() if f.lower().endswith(
            ('.png','.jpg','.jpeg','.tiff','.bmp')
        )]
        
        if not images:
            raise FileNotFoundError("No scientific plots found in directory")
            
        print("\nAvailable plots:")
        for i, img in enumerate(images):
            print(f"{i+1}. {img}")
            
        selection = int(input("\nSelect plot to analyze (number): ")) - 1
        image_path = images[selection]
        
        print(f"\n🚀 Analyzing {image_path} with GPU acceleration...")
        analysis = analyzer.analyze_plot(image_path)
        
        output_file = f"analysis_{os.path.splitext(image_path)[0]}.txt"
        with open(output_file, "w") as f:
            f.write(analysis)
            
        print(f"\n✅ Analysis complete. Results saved to {output_file}")
        print("\n=== ANALYSIS RESULTS ===\n")
        print(analysis)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
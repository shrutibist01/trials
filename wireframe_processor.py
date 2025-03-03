import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from crewai import Agent, Task, Crew, LLM
import os
import sys
import json
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Use Gemini API key instead of OpenAI
api_key = "AIzaSyCCQWxsZK6xhLHW8rXxiTULrRLpepFe4is"
print(api_key)
if not api_key:
    print("ERROR: GEMINI_API_KEY not found")
    sys.exit(1)
print(f"API key found (starts with: {api_key[:5]}...)")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Step 1: Image Processing to Extract Wireframe Elements
def process_wireframe(image_path):
    print(f"Processing image: {image_path}")

    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load img
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Failed to load image: {image_path}")
        raise ValueError(f"Failed to load image: {image_path}")
    
    print(f"Image loaded successfully. Dimensions: {img.shape}")
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        cv2.imwrite("detected_edges.png", edges)
        print("Edge detection complete. Saved to 'detected_edges.png'")
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                rectangles.append((x, y, w, h))
        
        print(f"Detected {len(rectangles)} rectangular components")
        mser = cv2.MSER_create()
        text_regions, _ = mser.detectRegions(gray)
        
        ui_components = {
            "navigation": [],
            "headers": [],
            "content_areas": [],
            "buttons": [],
            "input_fields": []
        }
        
        for rect in rectangles:
            x, y, w, h = rect
            if w > img.shape[1] * 0.8 and h < img.shape[0] * 0.15:
                ui_components["navigation"].append(rect)
            elif w < 150 and h < 50:
                ui_components["buttons"].append(rect)
            elif w < 300 and h < 40:
                ui_components["input_fields"].append(rect)
            elif h > 50 and w > 200:
                ui_components["content_areas"].append(rect)
        
        viz_img = img.copy()
        colors = {
            "navigation": (255, 0, 0),      
            "buttons": (0, 255, 0),        
            "input_fields": (0, 0, 255),    
            "content_areas": (255, 255, 0)  
        }
        
        for component_type, rects in ui_components.items():
            if component_type in colors:
                for (x, y, w, h) in rects:
                    cv2.rectangle(viz_img, (x, y), (x + w, y + h), colors[component_type], 2)
        
        cv2.imwrite("detected_components.png", viz_img)
        print("Component detection complete. Visualization saved to 'detected_components.png'")
  
        for component_type, rects in ui_components.items():
            print(f"Detected {len(rects)} {component_type}")
        
        return {
            "original_image": img,
            "edges": edges,
            "components": ui_components,
            "layout": {"width": img.shape[1], "height": img.shape[0]}
        }
        
    except Exception as e:
        print(f"ERROR during image processing: {str(e)}")
        raise
    
def setup_crew_agents():
    print("Setting up CrewAI agents...")
    try:
        # Use Gemini model instead of GPT
        gemini_model = "gemini-1.5-flash"
        
        # UI analyzer agent
        ui_analyzer = Agent(
            role="UI Analyzer",
            goal="Accurately interpret wireframe components and their relationships",
            backstory="Expert in UI/UX design with deep understanding of design patterns",
            verbose=True,
            llm=LLM(model=gemini_model, api_key=api_key)
        )
        
        # HTML developer agent
        html_developer = Agent(
            role="HTML Developer",
            goal="Convert wireframe components into semantic HTML structure",
            backstory="Senior front-end developer with focus on accessible, semantic HTML",
            verbose=True,
            llm=LLM(model=gemini_model, api_key=api_key)
        )
        
        # CSS stylist agent
        css_stylist = Agent(
            role="CSS Stylist",
            goal="Create responsive, clean CSS that matches the wireframe layout",
            backstory="CSS expert with experience in responsive design and modern CSS frameworks",
            verbose=True,
            llm=LLM(model=gemini_model, api_key=api_key)
        )
        
        print("CrewAI agents created successfully")
        return ui_analyzer, html_developer, css_stylist
        
    except Exception as e:
        print(f"ERROR setting up CrewAI agents: {str(e)}")
        raise

# Define CrewAI Tasks
def create_tasks(ui_analyzer, html_developer, css_stylist, wireframe_data):
    print("Creating CrewAI tasks...")
    try:
        with open("wireframe_data.json", "w") as f:
            serializable_data = {
                "components": wireframe_data["components"],
                "layout": wireframe_data["layout"]
            }
            json.dump(serializable_data, f, indent=2)
        
        wireframe_analysis_task = Task(
            description=f"""
            Analyze the wireframe data and identify all UI components, their hierarchy, and relationships.
            Layout dimensions: width={wireframe_data['layout']['width']}, height={wireframe_data['layout']['height']}
            
            Component Summary:
            - Navigation elements: {len(wireframe_data['components']['navigation'])}
            - Buttons: {len(wireframe_data['components']['buttons'])}
            - Input fields: {len(wireframe_data['components']['input_fields'])}
            - Content areas: {len(wireframe_data['components']['content_areas'])}
            
            Create a detailed analysis of the UI structure.
            """,
            agent=ui_analyzer,
            expected_output="JSON description of UI components with hierarchical structure and relationships"
        )
        
        html_generation_task = Task(
            description="Convert the UI component analysis into semantic HTML structure that follows accessibility best practices",
            agent=html_developer,
            expected_output="Complete HTML document with appropriate tags and structure",
            context=[wireframe_analysis_task]
        )
        
        css_generation_task = Task(
            description="Create responsive CSS that accurately represents the wireframe layout with appropriate styling",
            agent=css_stylist,
            expected_output="Complete CSS stylesheet with responsive design considerations",
            context=[wireframe_analysis_task, html_generation_task]
        )
        
        print("CrewAI tasks created successfully")
        return wireframe_analysis_task, html_generation_task, css_generation_task
        
    except Exception as e:
        print(f"ERROR creating CrewAI tasks: {str(e)}")
        raise

# Execute the Pipeline
def wireframe_to_code(image_path):
    try:
        print("\n--- STEP 1: Processing Wireframe Image ---")
        wireframe_data = process_wireframe(image_path)
        
        print("\n--- STEP 2: Setting Up CrewAI Agents ---")
        ui_analyzer, html_developer, css_stylist = setup_crew_agents()
        
        print("\n--- STEP 3: Creating Tasks ---")
        wireframe_analysis_task, html_generation_task, css_generation_task = create_tasks(
            ui_analyzer, html_developer, css_stylist, wireframe_data
        )
        
        print("\n--- STEP 4: Running CrewAI Process ---")
        crew = Crew(
            agents=[ui_analyzer, html_developer, css_stylist],
            tasks=[wireframe_analysis_task, html_generation_task, css_generation_task],
            verbose=True
        )
        
        result = crew.kickoff()
        print("CrewAI process complete!")
        
        # Verify results
        if "html_generation_task" not in result:
            print(f"WARNING: Expected 'html_generation_task' in results but got: {list(result.keys())}")

            html_content = str(result.get(list(result.keys())[0], ""))
            css_content = ""
            analysis_content = ""
        else:
            html_content = result["html_generation_task"]
            css_content = result["css_generation_task"] 
            analysis_content = result["wireframe_analysis_task"]
        
        return {
            "html": html_content,
            "css": css_content,
            "analysis": analysis_content
        }
        
    except Exception as e:
        print(f"ERROR in wireframe_to_code: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty results in case of error
        return {
            "html": f"<!-- Error occurred: {str(e)} -->",
            "css": f"/* Error occurred: {str(e)} */",
            "analysis": f"Error occurred: {str(e)}"
        }

if __name__ == "__main__":
    try:
        image_path = "wireframe.png"  
        
        print(f"Starting wireframe to code conversion for: {image_path}")
        result = wireframe_to_code(image_path)
        
        # Save results
        with open("output.html", "w") as f:
            f.write(result["html"])
        print(f"HTML file saved to 'output.html' ({len(result['html'])} characters)")
        
        with open("styles.css", "w") as f:
            f.write(result["css"])
        print(f"CSS file saved to 'styles.css' ({len(result['css'])} characters)")
        
        with open("analysis.json", "w") as f:
            f.write(result["analysis"])
        print(f"Analysis saved to 'analysis.json' ({len(result['analysis'])} characters)")
        
        # Display visualization
        try:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
            plt.title("Original Wireframe")
            
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.imread("detected_edges.png"), cmap='gray')
            plt.title("Edge Detection")
            
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(cv2.imread("detected_components.png"), cv2.COLOR_BGR2RGB))
            plt.title("Detected Components")
            
            plt.tight_layout()
            plt.savefig("wireframe_analysis.png")
            print("Visualization saved to 'wireframe_analysis.png'")
            plt.show()
        except Exception as vis_error:
            print(f"Visualization error (non-critical): {str(vis_error)}")
        
        print("Wireframe successfully converted to HTML/CSS!")
        print("Output files:")
        print("  - output.html: HTML code")
        print("  - styles.css: CSS styling")
        print("  - analysis.json: Component analysis")
        print("  - detected_edges.png: Edge detection visualization")
        print("  - detected_components.png: Component detection visualization")
        print("  - wireframe_analysis.png: Full analysis visualization")
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
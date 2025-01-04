from flask import Flask, render_template, request, jsonify, Response
from groq import Groq
from PIL import Image
import requests
import base64
import io
import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure API clients
groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
FLUX_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
FLUX_HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

# Store processing status
processing_status = {}

@app.route('/status/<session_id>')
def status(session_id):
    def generate():
        while True:
            # Get current status
            if session_id in processing_status:
                status_data = processing_status[session_id]
                yield f"data: {json.dumps(status_data)}\n\n"
                
                # If processing is complete, stop sending updates
                if status_data.get('complete', False):
                    processing_status.pop(session_id, None)
                    break
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

def encode_image(image_file):
    """Convert uploaded image to base64"""
    return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(base64_image):
    """Get detailed physical description using Groq's vision model"""
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Provide a detailed physical description of this person. Include: body type/build, estimated height, facial features, hair style/color, skin tone, and any distinctive features. Focus on objective physical characteristics that would be important for creating a superhero design. Be specific and detailed in your description."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=300
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

def generate_superhero_prompt(appearance_desc):
    """Generate detailed superhero transformation prompt"""
    try:
        superhero_prompt = f"""Create a unique superhero design that matches: {appearance_desc}
Requirements:
- Use their exact physical features (height, build, hair, face shape)
- Create a superhero costume that complements their appearance
- Show them in a dynamic action pose
- Include dramatic lighting effects
- High-detail comic book art style
- Make the costume design unique to their features
Art style: Modern comic book art, highly detailed digital illustration
Output: A full-body superhero portrait in dramatic pose
"""
        # For FLUX API, we'll add some additional style guidance
        enhanced_prompt = f"{superhero_prompt}\nStyle reference: High-quality digital art, detailed comic book illustration, dynamic composition, professional lighting, 4k, highly detailed"
        
        return enhanced_prompt
    except Exception as e:
        print(f"Error generating superhero prompt: {e}")
        return None

def generate_image(prompt):
    """Generate image using FLUX API"""
    try:
        response = requests.post(
            FLUX_API_URL,
            headers=FLUX_HEADERS,
            json={"inputs": prompt}
        )
        return response.content
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    session_id = str(time.time())  # Create unique session ID
    processing_status[session_id] = {'status': 'Starting...', 'progress': 0}
    
    try:
        # Stage 1: Convert image to base64
        processing_status[session_id] = {
            'status': 'Analyzing Image...',
            'progress': 25,
            'detail': 'Processing and analyzing your image...'
        }
        base64_image = encode_image(image_file)
        
        # Stage 2: Get image description
        processing_status[session_id] = {
            'status': 'Generating Design...',
            'progress': 50,
            'detail': 'Creating your unique superhero design...'
        }
        appearance_desc = analyze_image(base64_image)
        if not appearance_desc:
            raise Exception('Failed to analyze image')
        
        # Stage 3: Generate superhero prompt
        processing_status[session_id] = {
            'status': 'Transforming...',
            'progress': 75,
            'detail': 'Transforming into superhero...'
        }
        superhero_prompt = generate_superhero_prompt(appearance_desc)
        if not superhero_prompt:
            raise Exception('Failed to generate superhero prompt')
        
        # Stage 4: Generate final image
        processing_status[session_id] = {
            'status': 'Finalizing...',
            'progress': 90,
            'detail': 'Finalizing your superhero image...'
        }
        generated_image = generate_image(superhero_prompt)
        if not generated_image:
            raise Exception('Failed to generate superhero image')
        
        # Complete
        processing_status[session_id] = {
            'status': 'Complete!',
            'progress': 100,
            'detail': 'Transformation complete!',
            'complete': True
        }
        
        # Convert to base64 for frontend display
        generated_image_b64 = base64.b64encode(generated_image).decode('utf-8')
        
        return jsonify({
            'session_id': session_id,
            'generated_image': generated_image_b64
        })
        
    except Exception as e:
        error_message = str(e)
        processing_status[session_id] = {
            'status': 'Error',
            'progress': 0,
            'detail': error_message,
            'complete': True
        }
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
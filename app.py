from flask import Flask, request, jsonify, render_template
import query_data

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET'])
def root():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Handle chat requests with RAG enhancement"""
    data = request.json
    if not data or 'messages' not in data:
        return jsonify({"error": "Missing messages parameter"}), 400
    
    try:
        ollama_response = query_data.chat_with_rag(
            messages=data['messages'],
            model=data.get('model')
        )
        
        # Log entire raw response
        print("--- OLLAMA RESPONSE ---")
        import json
        print(json.dumps(ollama_response, indent=2))
        print("--- END RESPONSE ---")
        
        # Return the complete message object with role and content
        if 'message' in ollama_response and 'content' in ollama_response['message']:
            return jsonify({
                "message": {
                    "role": ollama_response['message']['role'],
                    "content": ollama_response['message']['content']
                }
            })
        else:
            return jsonify({"error": "Invalid response format from Ollama"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model": query_data.OLLAMA_MODEL,
        "embedding_model": "all-MiniLM-L6-v2",
        "collection": query_data.CHROMA_COLLECTION_NAME
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
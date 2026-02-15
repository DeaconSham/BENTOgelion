from flask import Flask, request, jsonify, render_template
import math

app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')

# Global storage
state = {
    "x": 0,
    "y": 0,
    "h": 0,
    "resources": [] # List of {x, y, type}
}

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for bot_logic.py to update movement
@app.route('/update_location', methods=['POST'])
def update_location():
    data = request.json
    state["x"] = data.get("x", 0)
    state["y"] = data.get("y", 0)
    state["h"] = data.get("h", 0)
    return jsonify({"status": "ok"})

# Add this at the top of app.py
def is_new_discovery(new_label, new_x, new_y, threshold=15.0):
    for item in state["resources"]:
        # Calculate distance between current position and previous discovery
        dist = math.sqrt((new_x - item['x'])**2 + (new_y - item['y'])**2)
        
        # If it's the same object and we haven't moved much, it's not "new"
        if item['label'] == new_label.upper() and dist < threshold:
            return False
    return True
# NEW: Endpoint for vision.py to report resources
@app.route('/resource_found', methods=['POST'])
def resource_found():
    data = request.json
    # Grab the 'label' from the vision script
    label_text = data.get("label", "Unknown Object").upper()
    curr_x = state["x"]
    curr_y = state["y"]
    if is_new_discovery(label_text, curr_x, curr_y):
        new_item = {
            "x": curr_x, 
            "y": curr_y,
            "label": label_text
        }
        state["resources"].append(new_item)
        print(f"NEW LOG: {label_text} at ({curr_x}, {curr_y})")
        return jsonify({"status": "added"})
    return jsonify({"status": "text mapped"})


# Endpoint for map.js to get everything at once
@app.route('/get_telemetry', methods=['GET'])
def get_telemetry():
    return jsonify(state)

if __name__ == "__main__":
    # Use 0.0.0.0 so you can access the map from your laptop IP
    app.run(host='0.0.0.0', port=5000)
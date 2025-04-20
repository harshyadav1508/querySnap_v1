import os
import datetime
import threading
import pyautogui
from flask import Flask, render_template, jsonify, Response
from pynput import keyboard
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import json
import queue  # Import the queue module

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Read API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key is missing! Set GEMINI_API_KEY in your environment variables.")
genai.configure(api_key=api_key)

# Folder to save screenshots inside static
SAVE_DIR = os.path.join('static', 'screenshots')
os.makedirs(SAVE_DIR, exist_ok=True)

# Shared variables to hold the latest screenshot filename and Gemini response
latest_screenshot = None
latest_solution = "Press Ctrl+Shift+CapsLock to get started." # Initial message

# --- NEW: Thread-safe queue for communicating solutions to SSE clients ---
solution_queue = queue.Queue()

def call_gemini_api(image_path):
    """Send image to Gemini and get Python solution."""
    global latest_solution
    try:
        # --- NEW: Signal loading state ---
        solution_queue.put({"type": "loading"})

        img = Image.open(image_path)
        # --- NOTE: Consider using a more recent/appropriate model if available ---
        # E.g., "gemini-1.5-flash" or "gemini-1.5-pro" might offer better results
        model = genai.GenerativeModel("gemini-2.0-flash") # Example: Using 1.5 flash
        prompt = """
        You are an expert Python programmer specializing in Data Structures and Algorithms (DSA).
        Analyze the provided image containing a programming question.
        Provide a comprehensive answer including the following steps:

        1.  **Problem Understanding:** Briefly re-state the problem in your own words.
        2.  **Brute Force Approach:**
            *   Explain the logic in clear bullet points.
            *   Provide the Python code implementation with comments.
            *   State the Time Complexity (e.g., O(n^2)).
            *   State the Space Complexity (e.g., O(1)).
        3.  **Optimized Approach:**
            *   Explain the logic using appropriate data structures or algorithms in clear bullet points.
            *   Provide the Python code implementation with comments.
            *   State the Time Complexity (e.g., O(n log n) or O(n)).
            *   State the Space Complexity (e.g., O(n) or O(1)).
        4.  **Potential Interview Follow-up Questions:**
            *   List 2-3 likely questions an interviewer might ask about your solutions (e.g., edge cases, constraints, alternative optimizations).
            *   Provide brief answers to these potential questions.

        **Formatting Instructions:**
        *   Use Markdown for formatting (bolding, code blocks, bullet points).
        *   Clearly label each section (Problem Understanding, Brute Force, Optimized, Follow-up Questions).
        *   Ensure Python code is correctly formatted within Markdown code blocks (```python ... ```).
        """
        # Increased timeout and retry configurations
        response = model.generate_content(
            [prompt, img],
            request_options={"timeout": 600} # Example: 10 minute timeout
            # Add generation_config for safety if needed, e.g.,
            # generation_config=genai.types.GenerationConfig(
            #     candidate_count=1,
            #     stop_sequences=['\n\n\n'], # Example stop sequence
            #     max_output_tokens=8192, # Max for gemini-1.5-flash
            #     temperature=0.7
            # )
        )

        # Check for blocked content (optional but good practice)
        if not response.candidates:
             solution_text = "Error: Response was blocked or empty."
             print(solution_text)
        # Check if response generation was stopped early due to safety or other reasons
        elif response.prompt_feedback.block_reason:
             solution_text = f"Error: Response blocked due to {response.prompt_feedback.block_reason.name}"
             print(solution_text)
        else:
            solution_text = response.text.strip()
            print("Gemini API response received.")

        latest_solution = solution_text
        # --- NEW: Put the final solution onto the queue ---
        solution_queue.put({"type": "solution", "content": latest_solution})

    except Exception as e:
        error_message = f"Error calling Gemini API: {e}"
        latest_solution = error_message
        print(error_message)
        # --- NEW: Put the error onto the queue ---
        solution_queue.put({"type": "error", "content": error_message})

def take_screenshot_and_process():
    """Take screenshot, save it, and send to Gemini API."""
    global latest_screenshot
    try:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"screenshot_{now}.png"
        filepath = os.path.join(SAVE_DIR, filename)

        # Ensure SAVE_DIR exists (it should, but double-check)
        os.makedirs(SAVE_DIR, exist_ok=True)

        image = pyautogui.screenshot()
        image.save(filepath)
        latest_screenshot = filename # Keep track for potential display if needed
        print(f"Screenshot saved to {filepath}")

        # Call Gemini API in a separate thread to avoid blocking Flask
        # Pass the queue reference if needed, but global works here
        threading.Thread(target=call_gemini_api, args=(filepath,), daemon=True).start()
    except Exception as e:
        error_msg = f"Error taking screenshot or starting thread: {e}"
        print(error_msg)
        # Optionally update latest_solution and push to queue if screenshot fails
        # latest_solution = error_msg
        # solution_queue.put({"type": "error", "content": error_msg})


# Hotkey keys
CTRL_KEYS = {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
SHIFT_KEYS = {keyboard.Key.shift_l, keyboard.Key.shift_r}
CAPSLOCK_KEY = keyboard.Key.caps_lock
current_keys = set()

def on_press(key):
    # Normalize key if it's a character key
    try:
        k = key.char
    except AttributeError:
        k = key # It's a special key

    current_keys.add(k) # Add the normalized key

    if k == keyboard.Key.esc:
        print("ESC pressed, exiting hotkey listener...")
        # --- IMPORTANT: Need to stop the listener cleanly ---
        # Returning False is the correct way for pynput
        return False

    # Check combination
    ctrl_pressed = any(k in current_keys for k in CTRL_KEYS)
    shift_pressed = any(k in current_keys for k in SHIFT_KEYS)
    capslock_pressed = CAPSLOCK_KEY in current_keys

    if ctrl_pressed and shift_pressed and capslock_pressed:
        print("Ctrl+Shift+CapsLock detected! Taking screenshot and processing...")
        # --- Execute in a separate thread to avoid blocking the listener ---
        threading.Thread(target=take_screenshot_and_process, daemon=True).start()


def on_release(key):
     # Normalize key if it's a character key
    try:
        k = key.char
    except AttributeError:
        k = key # It's a special key

    if k in current_keys:
        current_keys.remove(k)
    # Remove specific control/shift keys as well if needed
    if key in CTRL_KEYS and key in current_keys:
         current_keys.remove(key)
    if key in SHIFT_KEYS and key in current_keys:
         current_keys.remove(key)
    if key == CAPSLOCK_KEY and key in current_keys:
         current_keys.remove(key)


# --- Store the listener instance to stop it later ---
listener = None

def start_hotkey_listener():
    global listener
    # Use a non-blocking listener if preferred, but join() is fine in a thread
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        print("Hotkey listener started.")
        listener.join() # Blocks this thread until listener.stop() or on_press returns False
    print("Hotkey listener stopped.") # This will print when ESC is pressed


@app.route('/')
def home():
    # Pass the *current* latest solution for initial page load
    return render_template('index.html', initial_solution=latest_solution)

# This optional endpoint is less useful now with SSE, but can be kept for debugging
@app.route('/api/latest_solution')
def get_latest_solution():
    """API endpoint to fetch latest solution."""
    return jsonify({
        # "image": f'screenshots/{latest_screenshot}' if latest_screenshot else None, # Image not strictly needed for solution
        "solution": latest_solution
    })

@app.route('/stream_solution')
def stream_solution():
    def event_stream():
        # --- NEW: Loop indefinitely, waiting for items from the queue ---
        while True:
            try:
                # Wait for a new solution/status update from the queue
                # This blocks until an item is available
                data = solution_queue.get()
                print(f"SSE: Sending data to client: {data}") # Debug print
                # Format as Server-Sent Event
                # Ensure data is JSON serialized if it's a dictionary
                yield f"data: {json.dumps(data)}\n\n"
            except GeneratorExit:
                # Client disconnected
                print("SSE: Client disconnected")
                break # Exit the loop
            except Exception as e:
                print(f"SSE: Error in event stream: {e}")
                # Optionally send an error to the client
                error_data = {"type": "error", "content": f"Server stream error: {e}"}
                try:
                    yield f"data: {json.dumps(error_data)}\n\n"
                except Exception: # Handle cases where yield fails after error
                    pass
                # Depending on the error, you might want to break or continue
                # For now, let's continue listening for next item
    # Set content type for SSE
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    # Start hotkey listener in a separate daemon thread
    listener_thread = threading.Thread(target=start_hotkey_listener, daemon=True)
    listener_thread.start()

    print("Flask server starting on http://0.0.0.0:5000")
    print("Press Ctrl+Shift+CapsLock to take screenshot and get solution.")
    print("Press ESC in the window *where the script was run* to stop the hotkey listener.")
    # Use threaded=True for Flask dev server to handle multiple requests (like SSE + home) better
    # Use debug=False for production or when dealing with threads/SSE stability
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    # --- Optional: Attempt to stop listener if app exits ---
    # This might not always run if the app is killed forcefully
    if listener and listener.is_alive():
        print("Stopping listener...")
        listener.stop()

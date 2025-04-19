from multion.client import MultiOn
import os
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import multion
import time
from PIL import Image
from io import BytesIO
import requests
from IPython.display import display, Markdown, HTML, clear_output

def load_env():
    _ = load_dotenv(find_dotenv())

def get_multi_on_api_key():
    load_env()
    multi_on_api_key = os.getenv("MULTION_API_KEY")
    return multi_on_api_key

def get_multi_on_client():
    multi_on_api_key = get_multi_on_api_key()
    return MultiOn(api_key=multi_on_api_key)

class ImageUtils:
    @staticmethod
    def get_screenshot(screenshot_url):
        """Utility function to download and process screenshot

        Args:
            screenshot_url (str): URL of the screenshot to download

        Returns:
            PIL.Image or None: The screenshot as a PIL Image, or None if the download fails
        """
        if not screenshot_url or not screenshot_url.startswith("http"):
            return None
        try:
            response = requests.get(screenshot_url)
            if response.status_code != 200:
                return None
            img_bytes = response.content
            img_io = BytesIO(img_bytes)
            return Image.open(img_io)
        except:
            return None

def visualizeSession(response, max_image_width=800, clear_previous=False, max_message_height=400, show_screenshot=True, max_cell_height=600):
    """
    Display MultiOn session response in a well-formatted way in Jupyter notebook with all content contained
    within a scrollable cell.
    
    Args:
        response (SessionStepSuccess): The response object from multionClient.execute_task
        show_screenshot (bool, optional): Whether to display the screenshot. Defaults to True.
        max_image_width (int, optional): Maximum width for displayed screenshots in pixels. Defaults to 800.
        clear_previous (bool, optional): Whether to clear previous cell output before displaying. Defaults to False.
        max_message_height (int, optional): Maximum height for the scrollable message container in pixels. Defaults to 400.
        max_cell_height (int, optional): Maximum height for the overall cell container in pixels. Defaults to 600.
    """
    # Clear previous output if requested
    if clear_previous:
        clear_output(wait=True)
    
    # Determine status color
    status_color = {
        "CONTINUE": "blue",
        "DONE": "green",
        "ERROR": "red",
        "ASK_USER": "orange"
    }.get(response.status, "gray")
    
    # Process the message to ensure proper line breaks in markdown
    message_content = ""
    if hasattr(response, 'message') and response.message:
        # Replace any line breaks that aren't properly formatted for markdown
        message_md = response.message.replace('\n', '  \n')  # Adding two spaces before newline for markdown
        message_content = f'''
        <div style="max-height: {max_message_height}px; overflow-y: auto; border: 1px solid #e0e0e0; 
                    padding: 15px; border-radius: 5px; background-color: #f9f9f9;">
            {Markdown(message_md)._repr_markdown_()}
        </div>
        '''
    
    # Prepare screenshot HTML if needed
    screenshot_html = ""
    if show_screenshot and hasattr(response, 'screenshot') and response.screenshot:
        screenshot_html = '<h4>Screenshot</h4>'
        try:
            # Function to resize image while maintaining aspect ratio
            def resize_image(img, max_width=max_image_width):
                width, height = img.size
                if width > max_width:
                    ratio = max_width / width
                    new_height = int(height * ratio)
                    img = img.resize((max_width, new_height), Image.LANCZOS)
                return img
            
            # If screenshot is already a PIL Image
            if isinstance(response.screenshot, Image.Image):
                img = resize_image(response.screenshot)
                img_buffer = BytesIO()
                img.save(img_buffer, format="PNG")
                img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                screenshot_html += f'<img src="data:image/png;base64,{img_str}" />'
            # If screenshot is a URL
            elif isinstance(response.screenshot, str) and response.screenshot.startswith('http'):
                screenshot_html += f'<img src="{response.screenshot}" style="max-width: {max_image_width}px;" />'
            # If screenshot is base64 encoded
            elif isinstance(response.screenshot, str):
                try:
                    screenshot_html += f'<img src="data:image/png;base64,{response.screenshot}" style="max-width: {max_image_width}px;" />'
                except:
                    screenshot_html += "<p><em>Screenshot could not be decoded from base64</em></p>"
            else:
                screenshot_html += "<p><em>Screenshot in unsupported format</em></p>"
        except Exception as e:
            screenshot_html += f"<p><em>Error displaying screenshot: {str(e)}</em></p>"
    
    # Construct the full HTML with everything in a single scrollable container
    url_html = ""
    if hasattr(response, 'url') and response.url:
        url_html = f'<div style="padding: 10px; background-color: #f5f5f5; margin-bottom: 20px;"><strong>URL:</strong> <a href="{response.url}" target="_blank">{response.url}</a></div>'
    
    full_html = f'''
    <div style="max-height: {max_cell_height}px; overflow-y: auto; border: 1px solid #ddd; border-radius: 5px; padding: 10px;">
        <div style="padding: 10px; background-color: #f5f5f5; border-left: 5px solid {status_color}; margin-bottom: 20px;">
            <strong>Status:</strong> {response.status}
        </div>
        {url_html}
        <h4>Message</h4>
        {message_content}
        {screenshot_html}
    </div>
    '''
    
    # Display the combined container
    display(HTML(full_html))

def display_step_header(step_number):
    """
    Display a visually appealing step header in a Jupyter notebook.
    
    Args:
        step_number (int): The current step number to display
    """
    step_html = f'''
    <div style="
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 14px;
        margin: 15px 0px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        text-align: center;
    ">
        STEP {step_number}
    </div>
    '''
    display(HTML(step_html))


class SessionManager:
    """Manages browser sessions for the MultiOn client"""
    
    def __init__(self, base_url, multion_client):
        """Initialize the session manager with a base URL"""
        self.base_url = base_url
        self.current_url = base_url
        self.session_id = None
        self.screenshot = None
        self.last_response = None
        
        # Create the MultiOn client if not provided
        self.multion_client = multion_client if multion_client else MultiOnClient()
    
    def create_session(self):
        """Create a new browser session at the base URL"""
        session = self.multion_client.client.sessions.create(
            url=self.base_url, 
            include_screenshot=True
        )
        self.session_id = session.session_id
        self.multion_client.session_id = session.session_id
        self.current_url = session.url
        self.multion_client.current_url = session.url
        self.screenshot = session.screenshot
        self.multion_client.screenshot = session.screenshot
        return session
    
    def close_session(self):
        """Close the current browser session"""
        if self.session_id:
            self.multion_client.client.sessions.close(self.session_id)
            self.session_id = None
            self.multion_client.session_id = None
    
    def close_all_sessions(self):
        """Close all open browser sessions"""
        sessions = self.multion_client.client.sessions.list()
        for session in sessions.session_ids:
            self.multion_client.client.sessions.close(session)
    
    def navigate_to_url(self, url):
        """Navigate to a specific URL in the current session"""
        if not self.session_id:
            # If no session exists, create one at the specified URL
            session = self.multion_client.client.sessions.create(
                url=url, 
                include_screenshot=True
            )
            self.session_id = session.session_id
            self.multion_client.session_id = session.session_id
            self.current_url = session.url
            self.multion_client.current_url = session.url
            self.screenshot = session.screenshot
            self.multion_client.screenshot = session.screenshot
            return session
        
        # If a session exists, navigate to the URL
        response = self.multion_client.client.sessions.step(
            session_id=self.session_id,
            cmd=f"GO TO URL {url}",
            include_screenshot=True,
        )
        
        # Update state
        self.current_url = response.url
        self.multion_client.current_url = response.url
        self.screenshot = response.screenshot
        self.multion_client.screenshot = response.screenshot
        return response
    
    def execute_task(self, task):
        """Execute a task in the current browser session"""
        if not self.session_id:
            # Create a session if none exists
            self.create_session()
        
        # Execute the task
        response = self.multion_client.client.sessions.step(
            session_id=self.session_id,
            cmd=(
                "IMPORTANT: DO NOT ASK THE USER ANY QUESTIONS. "
                "All the necessary information is already provided and is on the current Page.\n"
                "Complete the task to the best of your abilities.\n\n"
                f"Task:\n\n{task}"
            ),
            include_screenshot=True,
        )
        
        # Update state
        self.last_response = response
        self.current_url = response.url
        self.multion_client.current_url = response.url
        self.screenshot = response.screenshot
        self.multion_client.screenshot = response.screenshot

        return response


class MultiOnDemo:
    """
    The main demo class for the MultiOn browser application.
    
    Students will focus on modifying this class for their lab.
    """
    
    def __init__(self, base_url, sessionManager, multion_client, instructions, action_engine=None):
        """
        Initialize the MultiOn Demo
        
        Args:
            base_url (str): The starting URL for the browser
            instructions (list): Example instructions to suggest
            action_engine (str, optional): The action engine to use
        """
        # Store the provided parameters
        self.base_url = base_url
        self.instructions = instructions
        self.action_engine = action_engine
        
        # Initialize state variables
        self.chat_history = []
        
        # Create the MultiOn client and session manager
        # self.multion_client = MultiOnClient()
        self.session_manager = sessionManager if sessionManager else SessionManager(base_url, multion_client)
        
        # Make sure we have an active session
        if not self.session_manager.session_id:
            self.session_manager.create_session()
    
    def process_url(self, url):
        """
        Process a URL change from the UI
        
        Args:
            url (str): The URL to navigate to
            
        Returns:
            tuple: (screenshot, current_url) for updating the UI
        """
        # STUDENTS MODIFY THIS: Navigate to the URL
        self.session_manager.navigate_to_url(url)
        
        # Return the updated screenshot and URL for the UI
        return ImageUtils.get_screenshot(self.session_manager.screenshot), self.session_manager.current_url
    
    def process_instruction(self, instruction):
        """
        Process a user instruction and update the UI
        
        This is a generator function that yields updates as processing occurs,
        allowing for real-time feedback.
        
        Args:
            instruction (str): The instruction to process
            
        Yields:
            tuple: (chat_history, screenshot, current_url, cleared_input) 
                   for updating the UI
        """
        # STUDENTS MODIFY THIS: This is where you'll implement your instruction
        # processing logic to interact with the MultiOn API
        
        # 1) Add the user instruction to chat history
        self.chat_history.append({"role": "user", "content": instruction})
        
        # 2) Add a temporary "Processing..." message
        self.chat_history.append({"role": "assistant", "content": "Processing..."})
        
        # 3) First yield to update UI immediately and clear input
        yield (
            self.chat_history,
            ImageUtils.get_screenshot(self.session_manager.screenshot),
            self.session_manager.current_url,
            gr.update(value="")  # Clear instruction box immediately
        )
        
        # 4) Process the instruction with potential multiple steps
        count = 0
        status = "CONTINUE"
        max_iterations = 5  # Limit iterations to prevent infinite loops
        
        while status == "CONTINUE" and count < max_iterations:
            # Execute the instruction
            response = self.session_manager.execute_task(instruction)
            # Remove the temporary "Processing..." message
            self.chat_history.pop()
            
            # Add the real response
            self.chat_history.append({"role": "assistant", "content": response.message})
            
            # Update status for next iteration
            status = response.status
            
            # If we need to continue, add another "Processing..." message
            if status == "CONTINUE" and count < max_iterations - 1:
                self.chat_history.append({"role": "assistant", "content": "Processing..."})
            
            # Yield the updated state
            yield (
                self.chat_history,
                ImageUtils.get_screenshot(self.session_manager.screenshot),
                self.session_manager.current_url,
                gr.update(value="")  # Keep input clear
            )
            
            # Increment counter and add a small delay
            count += 1
            time.sleep(1)
        
        # Final yield (if the last message is "Processing...", remove it)
        if self.chat_history[-1]["content"] == "Processing...":
            self.chat_history.pop()
            yield (
                self.chat_history,
                ImageUtils.get_screenshot(self.session_manager.screenshot),
                self.session_manager.current_url,
                gr.update(value="")
            )
    
    def create_demo(self):
        """Create and launch the Gradio UI"""
        with gr.Blocks() as demo:
            gr.Markdown("""
            # 🚀 MultiOn AI Browser
            ### Transforming web interactions into seamless Agentic Actions.
            """)

            with gr.Row():
                url_input = gr.Textbox(
                    value=self.session_manager.current_url,
                    label="Enter URL and press 'Enter' to load the page."
                )
                current_url_display = gr.Textbox(
                    value=self.session_manager.current_url,
                    label="Current URL",
                    interactive=False
                )

            with gr.Row():
                with gr.Column(scale=6):
                    chatbox = gr.Chatbot(type="messages")
                    instruction_input = gr.Textbox(label="Enter Instruction")
                    execute_btn = gr.Button("Execute")
                    gr.Examples(
                        examples=self.instructions,
                        inputs=instruction_input,
                        label="Suggested Instructions"
                    )
                with gr.Column(scale=4):
                    browser_display = gr.Image(
                        value=ImageUtils.get_screenshot(self.session_manager.screenshot),
                        label="Browser Screenshot",
                        interactive=False,
                        height=200,
                        width=300
                    )

            # Event bindings (Students should understand these)
            url_input.submit(
                fn=self.process_url,
                inputs=url_input,
                outputs=[browser_display, current_url_display]
            )
            instruction_input.submit(
                fn=self.process_instruction,
                inputs=instruction_input,
                outputs=[chatbox, browser_display, current_url_display, instruction_input]
            )
            execute_btn.click(
                fn=self.process_instruction,
                inputs=instruction_input,
                outputs=[chatbox, browser_display, current_url_display, instruction_input]
            )

        demo.launch(
            share=True,
            debug=True,
            inline=True,
            height=1000,
            width=800
        )

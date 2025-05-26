# #!/usr/bin/env python3

# ## @file terminal_commander.py
# #  @brief A ROS-enabled terminal command interpreter powered by an LLM.
# #  @details This script interprets user instructions using regex and a language model,
# #  launching terminal commands or ROS publishers accordingly.

# import subprocess  # Run terminal commands
# import os          # Interact with the OS (paths, environment)
# import re          # Use regular expressions to match patterns
# import sys         # Access system-level information
# import platform    # Detect OS platform
# import shlex       # Safely parse shell command strings
# import rospy       # ROS Python library
# from typing import Dict, Optional, List  # Type hints
# from llama_cpp import Llama              # Load local LLM
# from std_msgs.msg import String          # ROS message type

# ## @class TerminalCommander
# #  @brief Class for parsing and executing natural language terminal commands.
# class TerminalCommander:
    
#     ## @brief Initializes the LLM and system context.
#     def __init__(self):
#         """Initialize terminal commander with LLM"""
        
#         ## @var self.llm
#         #  Local LLaMA-based language model for understanding commands.
#         self.llm = Llama(
#             model_path="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
#             n_ctx=2048,
#             n_threads=8,
#             n_gpu_layers=40,
#             verbose=False
#         )
        
#         ## @var self.system_info
#         #  Basic OS and shell context.
#         self.system_info = {
#             "os": platform.system(),
#             "release": platform.release(),
#             "shell": os.environ.get('SHELL', '/bin/bash'),
#             "terminal": self._detect_terminal_emulator(),
#             "cwd": os.getcwd()
#         }
        
#         ## @var self.command_history
#         #  Stores previous user commands.
#         self.command_history = []
        
#         ## @var self.context_window
#         #  Optional context window for conversation memory.
#         self.context_window = []
        
#         ## @var self.terminal_launch_commands
#         #  Regex patterns for launching specific ROS commands.
#         self.terminal_launch_commands = [
#             r"^(?:please\s+)?(?:start|launch|run)\s+(?:the\s+)?(?:fake\s+)?publisher(?:\s+with|\s+message)?\s+(?P<message>.+)$",
#             r"^(?:please\s+)?(?:publish|send)\s+(?P<message>.+)\s+(?:to|on)\s+ros$"
#         ]

#     ## @brief Detects which terminal emulator is available.
#     #  @return Name of the terminal program.
#     def _detect_terminal_emulator(self) -> str:
#         """Detect system's default terminal"""
#         system = platform.system()
#         if system == "Linux":
#             for term in ["gnome-terminal", "konsole", "xfce4-terminal"]:
#                 if subprocess.run(["which", term], capture_output=True).returncode == 0:
#                     return term
#             return "xterm"
#         elif system == "Darwin":
#             return "Terminal.app"
#         return "xterm"

#     ## @brief Launch a new terminal window and run the specified command inside it.
#     #  @param command The shell command to execute.
#     def _launch_terminal_with_command(self, command: str):
#         """Launch terminal with command"""
#         terminal = self.system_info["terminal"]
#         try:
#             if platform.system() == "Linux":
#                 subprocess.Popen([terminal, "--", "bash", "-c", f"{command}; exec bash"])
#             elif platform.system() == "Darwin":
#                 subprocess.Popen(["osascript", "-e", f'tell app "Terminal" to do script "{command}"'])
#         except Exception as e:
#             print(f"Failed to launch terminal: {str(e)}")

#     ## @brief Parses user input and classifies it as a known or generic command.
#     #  @param user_input Natural language command from user.
#     #  @return Dict describing parsed action and its arguments.
#     def parse_command(self, user_input: str) -> Dict:
#         """Parse user input into commands"""
#         for pattern in self.terminal_launch_commands:
#             match = re.match(pattern, user_input.lower())
#             if match and match.group("message"):
#                 return {"action": "launch_publisher", "message": match.group("message").strip()}
#         return {"action": "execute", "command": user_input}

#     ## @brief Executes a command based on parsed action.
#     #  @param command_dict Dictionary containing action type and data.
#     def execute_command(self, command_dict: Dict):
#         """Execute parsed command"""
#         action = command_dict.get("action")
        
#         if action == "launch_publisher":
#             message = command_dict.get("message", "")
#             script_path = os.path.join(os.path.dirname(__file__), "fake_publisher.py")
#             cmd = f"python3 {script_path} '{message}'"
#             self._launch_terminal_with_command(cmd)
#             print(f"🚀 Launched publisher with message: '{message}'")

# ## @class ROSCommander
# #  @brief Extension of TerminalCommander with ROS messaging support.
# class ROSCommander(TerminalCommander):
    
#     ## @brief Initializes ROS node and publisher.
#     def __init__(self):
#         super().__init__()
#         rospy.init_node('ros_commander', anonymous=True)
        
#         ## @var self.publisher
#         #  Publishes command events to ROS topic.
#         self.publisher = rospy.Publisher('command_events', String, queue_size=10)
        
#     ## @brief Overrides base command execution to also publish to ROS.
#     #  @param command_dict Parsed user command dictionary.
#     def execute_command(self, command_dict: Dict):
#         """Override with ROS-specific execution"""
#         super().execute_command(command_dict)
#         self.publisher.publish(str(command_dict))

# ## @brief Entry point for the program. Initializes ROSCommander and listens for user input.
# if __name__ == "__main__":
#     commander = ROSCommander()
#     print("🤖 ROS Terminal Commander")
#     print("Enter commands like: 'start publisher with hello' or 'publish test to ROS'")
    
#     while not rospy.is_shutdown():
#         try:
#             user_input = input("\nROS $ ").strip()
#             if user_input.lower() in ("exit", "quit"):
#                 break
                
#             command = commander.parse_command(user_input)
#             commander.execute_command(command)
            
#         except KeyboardInterrupt:
#             print("\nUse 'exit' to quit")
#         except Exception as e:
#             print(f"Error: {str(e)}")

#!/usr/bin/env python3

## @file terminal_commander.py
#  @brief A ROS-enabled terminal command interpreter powered by an LLM.
#  @details This script automatically manages roscore and interprets user instructions using an optimized LLM.

import subprocess
import os
import re
import sys
import platform
import shlex
import rospy
import time
from typing import Dict, Optional, List
from llama_cpp import Llama
from std_msgs.msg import String

## @class TerminalCommander
#  @brief Class for parsing and executing natural language terminal commands.
class TerminalCommander:
    
    ## @brief Initializes the LLM and system context.
    def __init__(self):
        """Initialize terminal commander with optimized LLM"""
        
        # Optimized LLM configuration for faster response
        self.llm = Llama(
            model_path="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            n_ctx=1024,  # Reduced context for speed
            n_threads=max(1, os.cpu_count() - 1),  # Use most available cores
            n_gpu_layers=40 if 'CUDA_VISIBLE_DEVICES' in os.environ else 0,
            verbose=False
        )
        
        self.system_info = {
            "os": platform.system(),
            "release": platform.release(),
            "shell": os.environ.get('SHELL', '/bin/bash'),
            "terminal": self._detect_terminal_emulator(),
            "cwd": os.getcwd()
        }
        
        self.command_history = []
        self.context_window = []
        
        # Enhanced command patterns with mission understanding
        self.terminal_launch_commands = [
            r"^(?:please\s+)?(?:start|launch|run|begin)\s+(?:the\s+)?(?:fake\s+)?(?:publisher|mission)(?:\s+with|\s+message)?\s*(?P<message>.+)?$",
            r"^(?:please\s+)?(?:publish|send)\s+(?P<message>.+)\s+(?:to|on)\s+ros$",
            r"^(?:start|run)\s+mission\s+(?:with|using)\s+(?P<message>.+)$"
        ]
        
        # Track roscore process
        self.roscore_process = None

    ## @brief Detects which terminal emulator is available.
    def _detect_terminal_emulator(self) -> str:
        """Detect system's default terminal"""
        system = platform.system()
        if system == "Linux":
            for term in ["gnome-terminal", "konsole", "xfce4-terminal"]:
                if subprocess.run(["which", term], capture_output=True).returncode == 0:
                    return term
            return "xterm"
        elif system == "Darwin":
            return "Terminal.app"
        return "xterm"

    ## @brief Check if roscore is running
    def _is_roscore_running(self) -> bool:
        """Check if roscore is already running"""
        try:
            # Check master URI
            master_uri = os.environ.get('ROS_MASTER_URI', '')
            if not master_uri or 'localhost' not in master_uri:
                return False
                
            # Try to get published topics
            topics = rospy.get_published_topics()
            return True
        except:
            return False

    ## @brief Start roscore in a new terminal if not running
    def _ensure_roscore_running(self):
        """Ensure roscore is running before ROS operations"""
        if not self._is_roscore_running():
            print("🌐 Starting roscore...")
            if platform.system() == "Linux":
                self.roscore_process = subprocess.Popen(
                    [self.system_info["terminal"], "--", "bash", "-c", "roscore; exec bash"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            elif platform.system() == "Darwin":
                self.roscore_process = subprocess.Popen(
                    ["osascript", "-e", 'tell app "Terminal" to do script "roscore"'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Wait for roscore to initialize
            time.sleep(3)
            
            # Set ROS environment variables
            os.environ['ROS_MASTER_URI'] = 'http://localhost:11311'
            os.environ['ROS_HOSTNAME'] = 'localhost'

    ## @brief Launch a new terminal window with command
    def _launch_terminal_with_command(self, command: str):
        """Launch terminal with command"""
        terminal = self.system_info["terminal"]
        try:
            if platform.system() == "Linux":
                subprocess.Popen([terminal, "--", "bash", "-c", f"{command}; exec bash"])
            elif platform.system() == "Darwin":
                subprocess.Popen(["osascript", "-e", f'tell app "Terminal" to do script "{command}"'])
        except Exception as e:
            print(f"Failed to launch terminal: {str(e)}")

    ## @brief Use LLM to understand mission parameters
    def _understand_mission(self, message: str) -> str:
        """Use LLM to extract meaningful mission parameters"""
        prompt = f"""
        Extract the mission parameters from this user command:
        User: {message}
        
        Respond only with the extracted parameters in a single line.
        Example outputs:
        - "5"
        - "explore area 3"
        - "scan and map"
        """
        
        try:
            response = self.llm.create_completion(
                prompt,
                max_tokens=50,
                temperature=0.1,  # Low temperature for deterministic output
                stop=["\n"]
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"LLM processing error: {e}")
            return message  # Fallback to original message

    ## @brief Parse user input into commands
    def parse_command(self, user_input: str) -> Dict:
        """Parse user input into commands with enhanced understanding"""
        # First try direct pattern matching
        for pattern in self.terminal_launch_commands:
            match = re.match(pattern, user_input.lower())
            if match and (match.group("message") or 'mission' in user_input.lower()):
                message = match.group("message") or "default mission"
                processed_msg = self._understand_mission(message)
                return {
                    "action": "launch_publisher",
                    "message": processed_msg,
                    "original_input": user_input
                }
        
        # Fallback to LLM for complex commands
        return {"action": "execute", "command": user_input}

    ## @brief Execute parsed command
    def execute_command(self, command_dict: Dict):
        """Execute parsed command with ROS core management"""
        action = command_dict.get("action")
        
        if action == "launch_publisher":
            self._ensure_roscore_running()
            message = command_dict.get("message", "")
            script_path = os.path.join(os.path.dirname(__file__), "fake_publisher.py")
            cmd = f"python3 {script_path} '{message}'"
            self._launch_terminal_with_command(cmd)
            print(f"🚀 Mission started with parameters: '{message}'")

## @class ROSCommander
#  @brief Extension of TerminalCommander with ROS messaging support.
class ROSCommander(TerminalCommander):
    
    ## @brief Initializes ROS node and publisher.
    def __init__(self):
        super().__init__()
        try:
            rospy.init_node('ros_commander', anonymous=True)
            self.publisher = rospy.Publisher('command_events', String, queue_size=10)
        except rospy.ROSException as e:
            print(f"ROS initialization error: {e}")
            sys.exit(1)
        
    ## @brief Override with ROS-specific execution
    def execute_command(self, command_dict: Dict):
        """Override with ROS-specific execution"""
        super().execute_command(command_dict)
        try:
            self.publisher.publish(str(command_dict))
        except rospy.ROSInterruptException as e:
            print(f"ROS publish error: {e}")

## @brief Cleanup function
def cleanup():
    """Cleanup resources"""
    print("\n🛑 Shutting down ROS Commander...")
    if hasattr(rospy, 'is_shutdown') and not rospy.is_shutdown():
        rospy.signal_shutdown("User exit")

## @brief Main entry point
if __name__ == "__main__":
    commander = ROSCommander()
    print("🤖 ROS Terminal Commander - Enhanced Edition")
    print("Type commands like: 'start mission with 5' or 'publish test to ROS'")
    print("The system will automatically manage roscore when needed\n")
    
    try:
        while not rospy.is_shutdown():
            try:
                user_input = input("\nROS $ ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                    
                if user_input:
                    command = commander.parse_command(user_input)
                    commander.execute_command(command)
                    
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                print(f"Error: {str(e)}")
    finally:
        cleanup()
#!/usr/bin/env python3
"""
@file terminal_commander.py
@brief AI-powered terminal command processor with ROS support
@details This script provides natural language to terminal command translation,
         with special handling for ROS commands and terminal launching.
"""

import subprocess
import os
import re
import sys
import platform
import shlex
from typing import Dict, Optional, List
from llama_cpp import Llama

class TerminalCommander:
    """
    @class TerminalCommander
    @brief Main class for processing natural language terminal commands
    @details Handles command parsing, execution, and provides AI-powered assistance
    """
    
    def __init__(self):
        """
        @brief Constructor initializing the command processor
        @details Sets up the AI model, system information, and command patterns
        """
        # Initialize the LLM model with optimized settings
        self.llm = Llama(
            model_path="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            n_ctx=2048,        # Context window size
            n_threads=8,       # CPU threads to use
            n_gpu_layers=40,   # GPU layers for acceleration
            verbose=False      # Disable verbose output
        )
        
        # Gather system information for context-aware processing
        self.system_info = {
            "os": platform.system(),                # Operating system name
            "release": platform.release(),         # OS version
            "shell": os.environ.get('SHELL', '/bin/bash'),  # Default shell
            "terminal": self._detect_terminal_emulator(),   # Detected terminal
            "cwd": os.getcwd()                     # Current working directory
        }
        
        # Command history storage
        self.command_history: List[str] = []
        
        # Context window for LLM prompts
        self.context_window: List[str] = []
        
        # Patterns for detecting terminal launch requests
        self.terminal_launch_commands = [
            r"^(?:please\s+)?(?:can\s+you\s+)?(?:launch|start|open)\s+(?:a\s+)?terminal(?:\s+with|\s+running)?\s+(?P<command>.+)$",
            r"^(?:please\s+)?(?:can\s+you\s+)?(?:run|execute)\s+(?P<command>.+)\s+(?:in\s+)?(?:a\s+)?new\s+terminal$",
            r"^(?P<command>.+)\s+(?:in\s+)?(?:a\s+)?new\s+terminal$",
            r"^(?:please\s+)?(?:launch|start)\s+(?P<command>.+)$"
        ]
        
        # ROS-specific commands that should always launch in terminal
        self.ros_commands = {
            r"rostopic\s+list": "rostopic list",  # Standard rostopic command
            r"roscore": "roscore",                  # ROS core command
            r"roslaunch\s+.+": lambda x: x.group(0) if x else None  # ROS launch pattern
        }
        
        # Patterns for dangerous commands that should be blocked
        self.dangerous_patterns = [
            r"rm\s+-[rf]\s+",      # Force recursive deletion
            r":(){:|:&};:",        # Fork bomb
            r"mv\s+.*\s+/",        # Moving files to root
            r"dd\s+if=.*\s+of=",   # Disk overwrite
            r"chmod\s+[0-7]{3,4}\s+",  # Permission changes
            r">\s+/dev/sd[a-z]",  # Disk device writing
            r"mkfs",               # Filesystem creation
            r"fdisk",              # Partition editing
            r"\.\/\.\/\.\/"        # Path traversal
        ]
        
        # Natural language to command aliases
        self.command_aliases = {
            "list": "ls",          # List files
            "show": "cat",         # Display file
            "display": "cat",      # Display file
            "find": "grep",        # Search files
            "search": "grep",      # Search files
            "create": "touch",     # Create file
            "make": "mkdir",       # Create directory
            "remove": "rm",        # Delete file
            "delete": "rm",        # Delete file
            "copy": "cp",          # Copy file
            "move": "mv",          # Move file
            "change": "chmod",     # Change permissions
            "launch": "start",     # Start program
            "open": "start"        # Start program
        }

        # Help system categories and examples
        self.help_categories = {
            "Terminal Operations": {
                "Launch": [
                    "launch terminal with rostopic list",
                    "start roscore in new terminal",
                    "roslaunch package file.launch in new terminal"
                ],
                "Special": [
                    "rostopic list (auto-detected)",
                    "roscore (auto-detected)"
                ]
            },
            "Basic File Operations": {
                "List files": ["show files", "list hidden files"],
                "Navigate": ["go to documents", "move up one directory"],
                "Create": ["make file notes.txt", "create folder projects"],
                "Delete": ["remove old.log", "delete temp folder"],
                "Copy/Move": ["copy config.txt to backup", "move report.pdf to Documents"]
            }
        }

    def _detect_terminal_emulator(self) -> str:
        """
        @brief Detect the system's default terminal emulator
        @return String with terminal name
        @details Checks common terminals on Linux, defaults to system terminal on macOS/Windows
        """
        system = platform.system()
        if system == "Linux":
            # Check for common Linux terminals
            for term in ["gnome-terminal", "konsole", "xfce4-terminal", "alacritty", "kitty"]:
                if subprocess.run(["which", term], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                    return term
            return "xterm"  # Fallback terminal
        elif system == "Darwin":
            return "Terminal.app"  # macOS default
        elif system == "Windows":
            return "cmd.exe"  # Windows command prompt
        return "xterm"  # Ultimate fallback

    def _is_terminal_launch_request(self, user_input: str) -> bool:
        """
        @brief Check if input requests a new terminal launch
        @param user_input The natural language command string
        @return Boolean indicating if terminal launch is requested
        @details Checks both ROS commands and general terminal launch patterns
        """
        lower_input = user_input.lower()
        
        # First check for ROS commands
        for pattern in self.ros_commands:
            if re.search(pattern, lower_input):
                return True
                
        # Then check general terminal launch patterns
        for pattern in self.terminal_launch_commands:
            if re.match(pattern, lower_input):
                return True
        return False

    def _parse_terminal_launch_command(self, user_input: str) -> Dict:
        """
        @brief Parse terminal launch command from user input
        @param user_input The natural language command string
        @return Dictionary with action and command details
        @details Handles both ROS-specific and general terminal launch commands
        """
        lower_input = user_input.lower()
        
        # Handle ROS commands first
        for pattern, cmd in self.ros_commands.items():
            match = re.search(pattern, lower_input)
            if match:
                if callable(cmd):
                    command = cmd(match)
                else:
                    command = cmd
                if command:
                    return {"action": "launch_terminal", "command": command}
        
        # Handle general terminal launch patterns
        for pattern in self.terminal_launch_commands:
            match = re.match(pattern, lower_input)
            if match and match.group("command"):
                command = match.group("command").strip()
                return {"action": "launch_terminal", "command": command}
        
        # Default to direct execution if no match
        return {"action": "execute", "command": ""}

    def _launch_terminal_with_command(self, command: str):
        """
        @brief Launch new terminal with specified command
        @param command The command to execute in new terminal
        @details Handles platform-specific terminal launching with proper command formatting
        """
        system = platform.system()
        terminal = self.system_info["terminal"]
        
        try:
            # Special handling for ROS commands
            if any(cmd in command for cmd in ["rostopic", "roscore", "roslaunch"]):
                final_command = command
            else:
                # Include directory change for non-ROS commands
                final_command = f"cd {shlex.quote(self.system_info['cwd'])}; {command}"
            
            print(f"🚀 Launching new terminal with command: {command}")
            
            # Platform-specific terminal launching
            if system == "Linux":
                if terminal in ["gnome-terminal", "xfce4-terminal"]:
                    subprocess.Popen([terminal, "--", "bash", "-c", final_command + "; exec bash"])
                elif terminal == "konsole":
                    subprocess.Popen([terminal, "-e", "bash", "-c", final_command + "; exec bash"])
                else:
                    subprocess.Popen([terminal, "-e", "bash", "-c", final_command])
            elif system == "Darwin":
                # AppleScript for macOS Terminal
                script = f'''tell application "Terminal"
                    do script "{final_command}"
                    activate
                end tell'''
                subprocess.Popen(["osascript", "-e", script])
            elif system == "Windows":
                subprocess.Popen(["start", "cmd", "/k", final_command], shell=True)
            else:
                print(f"Terminal launch not supported on {system}")
        except Exception as e:
            print(f"Failed to launch terminal: {str(e)}")

    def _get_system_context(self) -> str:
        """
        @brief Generate system context string for LLM prompts
        @return Formatted context string
        @details Includes OS, shell, terminal, and recent command history
        """
        return f"""
        System Context:
        - OS: {self.system_info['os']} {self.system_info['release']}
        - Shell: {self.system_info['shell']}
        - Terminal: {self.system_info['terminal']}
        - Current Directory: {self.system_info['cwd']}
        - Recent Commands: {self.command_history[-3:] if self.command_history else 'None'}
        """

    def _is_dangerous(self, command: str) -> bool:
        """
        @brief Check if command matches dangerous patterns
        @param command The terminal command to check
        @return Boolean indicating if command is dangerous
        @details Compares against known dangerous command patterns
        """
        lower_cmd = command.lower()
        for pattern in self.dangerous_patterns:
            if re.search(pattern, lower_cmd):
                return True
        return False

    def parse_command(self, user_input: str) -> Dict:
        """
        @brief Parse natural language into executable command
        @param user_input Natural language command string
        @return Dictionary with action and command details
        @details Handles special cases before using LLM for translation
        """
        # Check for terminal launch requests first
        if self._is_terminal_launch_request(user_input):
            return self._parse_terminal_launch_command(user_input)
            
        # Handle help requests
        if user_input.lower() == "help":
            return {"action": "help", "category": None}
        elif user_input.lower().startswith("help "):
            return {"action": "help", "category": user_input[5:].strip()}
        
        # Direct command execution (with ! prefix)
        if user_input.startswith('!'):
            return {"action": "direct", "command": user_input[1:]}
        
        # Generate LLM prompt with system context
        context = self._get_system_context()
        prompt = f"""{context}
Convert this natural language request to a terminal command. 
ONLY respond with the command itself, no explanation or formatting.

Examples:
Request: "show me files in this directory"
Command: ls -l

Request: "search for python in all files"
Command: grep -r "python" .

Request: "{user_input}"
Command:"""
        
        # Get LLM response
        response = self.llm(
            prompt,
            max_tokens=64,      # Limit response length
            temperature=0.2,    # Low creativity for accuracy
            stop=["\n"]         # Stop at newline
        )
        
        # Process raw command from LLM
        raw_command = response["choices"][0]["text"].strip()
        
        # Apply command aliases if needed
        first_word = raw_command.split()[0]
        if first_word in self.command_aliases:
            raw_command = raw_command.replace(first_word, self.command_aliases[first_word], 1)
        
        return {"action": "execute", "command": raw_command}

    def execute_command(self, command_dict: Dict):
        """
        @brief Execute parsed command
        @param command_dict Dictionary containing action and command
        @details Handles terminal launches, direct execution, and help requests
        """
        action = command_dict.get("action")
        
        # Handle terminal launch requests
        if action == "launch_terminal":
            command = command_dict.get("command", "")
            if not command:
                print("No command specified for terminal launch")
                return
                
            self._launch_terminal_with_command(command)
            self.command_history.append(f"launch terminal with {command}")
            return
            
        # Handle help requests
        if action == "help":
            self.show_help(command_dict.get("category"))
            return
        
        command = command_dict.get("command", "")
        
        if not command:
            print("No command to execute")
            return
        
        # Safety check for dangerous commands
        if self._is_dangerous(command):
            print(f"⚠️  Dangerous command blocked: {command}")
            return
        
        # Add to command history
        self.command_history.append(command)
        
        try:
            # Special handling for history commands
            if command == "history":
                for i, cmd in enumerate(self.command_history, 1):
                    print(f"{i}: {cmd}")
                return
            elif command.startswith("repeat "):
                cmd_num = int(command.split()[1])
                if 1 <= cmd_num <= len(self.command_history):
                    command = self.command_history[cmd_num-1]
                else:
                    print("Invalid command number")
                    return
            
            print(f"Executing: {command}")
            
            # Handle shell features (pipes, redirection)
            if "|" in command or ">" in command or "<" in command:
                process = subprocess.Popen(command, shell=True, executable=self.system_info['shell'])
                process.wait()
            else:
                # Execute simple commands directly
                args = shlex.split(command)
                if args:
                    result = subprocess.run(args, capture_output=True, text=True)
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print(result.stderr, file=sys.stderr)
        except Exception as e:
            print(f"Command failed: {str(e)}")

    def explain_command(self, command: str):
        """
        @brief Explain what a terminal command will do
        @param command The terminal command to explain
        @details Uses LLM to generate human-readable explanation
        """
        prompt = f"""Explain exactly what this terminal command will do in 1-2 sentences:
Command: {command}
Explanation: This command will"""
        
        response = self.llm(
            prompt,
            max_tokens=100,     # Sufficient length for explanation
            temperature=0.3     # Balanced creativity/accuracy
        )
        
        explanation = "This command will" + response["choices"][0]["text"]
        print(f"🔍 Explanation: {explanation}")

    def suggest_improvement(self, command: str):
        """
        @brief Suggest safer/better alternative to command
        @param command The terminal command to improve
        @details Uses LLM to generate safer or more efficient alternatives
        """
        prompt = f"""Suggest a safer or more efficient alternative to this command:
Original: {command}
Alternative:"""
        
        response = self.llm(
            prompt,
            max_tokens=64,      # Limit to one command
            temperature=0.4,    # Slightly creative
            stop=["\n"]         # Stop at newline
        )
        
        alternative = response["choices"][0]["text"].strip()
        if alternative:
            print(f"💡 Suggestion: {alternative}")

    def show_help(self, category: Optional[str] = None):
        """
        @brief Display help information
        @param category Optional help category to show
        @details Shows general help or specific category examples
        """
        print("\n🖥️  Terminal Commander AI - Complete Help System")
        print("Type natural language commands or prefix with ! for direct execution")
        print("Special commands: history, explain <cmd>, suggest <cmd>, help [category]")
        
        if category:
            found = False
            print(f"\n🔎 Help for '{category}':")
            for cat_name, commands in self.help_categories.items():
                if category.lower() in cat_name.lower():
                    found = True
                    print(f"\n=== {cat_name.upper()} ===")
                    for cmd_type, examples in commands.items():
                        print(f"\n{cmd_type}:")
                        for example in examples:
                            print(f"  • {example}")
            if not found:
                print(f"No category found matching '{category}'. Available categories:")
                print(", ".join(self.help_categories.keys()))
        else:
            print("\n📚 Available Help Categories:")
            for i, category_name in enumerate(self.help_categories.keys(), 1):
                print(f"{i}. {category_name}")
            
            print("\nℹ️  Type 'help <category>' for detailed examples (e.g., 'help Terminal Operations')")

if __name__ == "__main__":
    """
    @brief Main entry point for Terminal Commander
    @details Initializes the commander and starts the interactive loop
    """
    commander = TerminalCommander()
    print("🖥️  Terminal Commander AI - Enhanced ROS Terminal Launcher")
    print("Try commands like: 'rostopic list' or 'please launch roscore'")
    print("Type 'help' for complete documentation")
    
    # Main interactive loop
    while True:
        try:
            user_input = input("\n$ ").strip()
            if not user_input:
                continue
                
            # Exit condition
            if user_input.lower() in ("exit", "quit"):
                break
                
            # Handle explanation requests
            if user_input.startswith("explain "):
                commander.explain_command(user_input[8:])
                continue
            # Handle suggestion requests
            elif user_input.startswith("suggest "):
                commander.suggest_improvement(user_input[8:])
                continue
                
            # Parse and execute command
            if user_input.startswith('!'):
                command = {"action": "direct", "command": user_input[1:]}
            else:
                command = commander.parse_command(user_input)
                
            commander.execute_command(command)
            
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except Exception as e:
            print(f"Error: {str(e)}")
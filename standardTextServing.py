import json
import os
import logging
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
from model_config import ModelConfig

def get_system_prompt() -> str:
    """Return the default system prompt for the assistant."""
    return "You are a master storyteller with a vivid imagination and a knack for crafting realistic yet random stories about people. Your goal is to create engaging, detailed, and believable narratives about fictional individuals, their lives, and their adventures. Each story should feel authentic, with rich descriptions, unique characters, and unexpected twists. Avoid making the stories too predictable or overly fantastical; instead, focus on grounding them in reality with nuance to their decisions. They can ultimately be rather boring, tragic, uneventful, or excited since there are many possible realisitc stories."

def get_default_sampling_params(config: Optional[ModelConfig] = None) -> SamplingParams:
    if config is None:
        config = ModelConfig()
    return SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

def initialize_model(model_id: str, config: Optional[ModelConfig] = None) -> LLM:
    """Initialize LLM with specified model and configurable settings."""
    if config is None:
        config = ModelConfig()
    try:
        model = LLM(
            model=model_id,
            trust_remote_code=True,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_util,
            tensor_parallel_size=config.tensor_parallel,
            dtype=config.dtype,
            enforce_eager=True
        )
        return model
    except Exception as e:
        logging.error(f"Error initializing model: {str(e)}")
        print("Ensure you have the correct model weights downloaded and CUDA is available")
        raise

class VLLMChatClient:
    def __init__(self, model_path: str, config: Optional[ModelConfig] = None, history_file: str = "chat_history.json"):
        """Initialize chat client with model and history file."""
        self.config = config or ModelConfig()
        self.model = initialize_model(model_path, self.config)
        self.history_file = history_file
        self.chat_histories = self._load_histories()

    def _load_histories(self) -> Dict[str, List[Dict[str, str]]]:
        """Load and validate chat histories from file."""
        if not os.path.exists(self.history_file):
            return {}
            
        try:
            with open(self.history_file, 'r') as f:
                histories = json.load(f)
            return {
                process_name: self._validate_history(history)
                for process_name, history in histories.items()
            }
        except json.JSONDecodeError:
            return {}

    def _validate_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Ensure history has correct format and system prompt."""
        if not isinstance(history, list) or not all(isinstance(msg, dict) for msg in history):
            return self._create_fresh_history()
            
        if not history or history[0].get('role') != 'system':
            history.insert(0, {"role": "system", "content": get_system_prompt()})
            
        return history

    def _create_fresh_history(self) -> List[Dict[str, str]]:
        """Create new history with system prompt."""
        return [{"role": "system", "content": get_system_prompt()}]
    
    def set_system_prompt(self, process_name: int, new_prompt: str) -> None:
        """
        Set a new system prompt for a specific process.

        If the process does not have an existing chat history, a new one is created with the system prompt as the first message.
        This method will always overwrite the first message in the history (the system prompt) for the given process.
        """
        process_name_str = str(process_name)
        if process_name_str not in self.chat_histories:
            self.chat_histories[process_name_str] = self._create_fresh_history()
        self.chat_histories[process_name_str][0] = {"role": "system", "content": new_prompt}
        self._save_histories()

    def get_chat_history(self, process_name: int, filter_role: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Get or create chat history for process. Optionally filter for a specific role (e.g., 'user', 'assistant', 'system').
        
        WARNING: Ensure you use a valid role string ('user', 'assistant', 'system', etc.) for filtering, otherwise the result may be empty.
        """
        process_name_str = str(process_name)
        if process_name_str not in self.chat_histories:
            self.chat_histories[process_name_str] = self._create_fresh_history()
        
        history = self.chat_histories[process_name_str]
        if filter_role is not None:
            history = [entry for entry in history if entry.get("role") == filter_role]
        return history

    def add_message(self, process_name: int, content: str, is_assistant: bool = False) -> None:
        """Add a message to process's chat history."""
        history = self.get_chat_history(process_name)
        history.append({
            "role": "assistant" if is_assistant else "user",
            "content": content
        })
        self._save_histories()

    def _save_histories(self) -> None:
        """Save chat histories to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.chat_histories, f, indent=2)
            logging.info(f"Chat histories saved to {self.history_file}")
        except Exception as e:
            logging.error(f"Failed to save chat histories to {self.history_file}: {str(e)}")
            raise

    def _clear_user_history(self, process_name: int) -> None:
        """Clear chat history for a process."""
        process_name_str = str(process_name)
        if process_name_str in self.chat_histories:
            del self.chat_histories[process_name_str]
            try:
                self._save_histories()
                logging.info(f"Cleared chat history for process name {process_name}")
            except Exception as e:
                logging.error(f"Failed to save updated chat histories after clearing process name {process_name}: {str(e)}")
                raise
        else:
            logging.warning(f"No history found for process name {process_name}")

    def clear_user_chat(self, process_name: int) -> str:
        """Clear chat history for a process and return a confirmation message."""
        try:
            self._clear_user_history(process_name)
            confirmation_message = f"Chat history for process name {process_name} has been cleared."
            logging.info(confirmation_message)
            return confirmation_message
        except Exception as e:
            error_message = f"Error clearing chat history for process name {process_name}: {str(e)}"
            logging.error(error_message)
            raise RuntimeError(error_message)

    def force_load_profiles(self, file_path: str, process_name: int) -> None:
        """
        Force load profiles from a JSON file into chat history.
        
        Assumptions:
        - JSON file has a top-level key 'compressed_profiles'
        - Each profile has fields: name, age, hobbies, work type, behavioral attributes
        - Function will overwrite any existing chat history for the given process_name
        - Profiles will be formatted as user messages in the chat history
        
        Args:
            file_path: Path to the JSON file containing profiles
            process_name: Process name to store the profiles under
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Clear existing history
            self._clear_user_history(process_name)
            
            # Create new history with system prompt
            self.chat_histories[str(process_name)] = self._create_fresh_history()
            
            # Convert each profile to a formatted string and add as user message
            for profile in data.get('compressed_profiles', []):
                profile_text = (
                    f"Name: {profile.get('name', 'Unknown')}\n"
                    f"Age: {profile.get('age', 'Unknown')}\n"
                    f"Occupation: {profile.get('work type', 'Unknown')}\n"
                    f"Hobbies: {', '.join(profile.get('hobbies', []))}\n"
                    f"Traits: {', '.join(profile.get('behavioral attributes', []))}"
                )
                self.add_message(process_name, profile_text)
                
            self._save_histories()
            logging.info(f"Successfully loaded {len(data.get('compressed_profiles', []))} profiles into process {process_name}")
            
        except Exception as e:
            error_message = f"Failed to load profiles from {file_path}: {str(e)}"
            logging.error(error_message)
            raise RuntimeError(error_message)

    def chat_request(self, process_name: int, message: str, sampling_params: Optional[SamplingParams] = None) -> str:
        """Process a user message and get model response."""
        self.add_message(process_name, message)
        chat_history = self.get_chat_history(process_name)

        # Use provided sampling parameters or default ones
        sampling_params = sampling_params or get_default_sampling_params(self.config)

        try:
            output = self.model.chat(messages=chat_history, sampling_params=sampling_params)

            if not output or not output[0]:
                raise ValueError("Model returned no output")

            response = output[0].outputs[0].text
            self.add_message(process_name, response, is_assistant=True)
            logging.info(f"Model response for process name {process_name}: {response}")
            return response

        except Exception as e:
            error_message = f"Failed to get model response for process name {process_name}: {str(e)}"
            logging.error(error_message)
            raise RuntimeError(error_message)
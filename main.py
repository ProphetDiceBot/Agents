import asyncio
import os
import json
import logging
import re
from typing import Dict, Any, List
import spacy
from dotenv import load_dotenv

# Updated import: ChatOpenAI now comes from langchain.chat_models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
from pydantic import SecretStr
from browser_use import Agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
MEMORY_FILE: str = "model_memory.json"

# Load spaCy NLP model (ensure you have run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")


class LLMSelector:
    def __init__(self, gemini_api_key: str, mistral_api_key: str) -> None:
        self.llms: Dict[str, Any] = {
            "gemini": ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp", 
                api_key=SecretStr(gemini_api_key)
            ),
            "mistral": ChatOpenAI(
                model="mistral-large", 
                openai_api_key=SecretStr(mistral_api_key)
            ),
        }
        self.memory: Dict[str, Dict[str, int]] = self.load_memory()

    def load_memory(self) -> Dict[str, Dict[str, int]]:
        """Load learned model selection memory from disk."""
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    memory = json.load(f)
                    logger.info("Memory loaded successfully.")
                    return memory
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
        return {}

    def save_memory(self) -> None:
        """Save updated model selection memory to disk."""
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(self.memory, f, indent=4)
            logger.info("Memory saved successfully.")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def update_memory(self, task: str, selected_model: str) -> None:
        """Update memory counts based on task and selected model."""
        doc = nlp(task.lower())
        for token in doc:
            if token.pos_ == "VERB":
                verb = token.lemma_
                if verb not in self.memory:
                    self.memory[verb] = {"gemini": 0, "mistral": 0}
                self.memory[verb][selected_model] += 1
                logger.debug(f"Updated memory for verb '{verb}': {self.memory[verb]}")

    def classify_task(self, task: str) -> Any:
        """
        Determines the best LLM for a given task.
        First uses learned memory; if unavailable, falls back to static rules.
        """
        doc = nlp(task.lower())
        verbs: List[str] = [token.lemma_ for token in doc if token.pos_ == "VERB"]

        # Use memory-based votes if available.
        votes = {"gemini": 0, "mistral": 0}
        for verb in verbs:
            if verb in self.memory:
                votes["gemini"] += self.memory[verb].get("gemini", 0)
                votes["mistral"] += self.memory[verb].get("mistral", 0)

        if votes["gemini"] != votes["mistral"]:
            selected = "gemini" if votes["gemini"] > votes["mistral"] else "mistral"
            logger.info(f"Memory-based vote: {votes} => Selecting {selected}")
            return self.llms[selected]

        # Fallback to static rules.
        action_keywords = ["search", "click", "browse", "scrape"]
        reasoning_keywords = ["find", "analyze", "interpret", "summarize", "comment"]
        lower_task = task.lower()
        if any(word in lower_task for word in action_keywords):
            logger.info("Static rule: Action keywords found, selecting mistral.")
            return self.llms["mistral"]
        elif any(word in lower_task for word in reasoning_keywords):
            logger.info("Static rule: Reasoning keywords found, selecting gemini.")
            return self.llms["gemini"]

        logger.info("Defaulting to gemini.")
        return self.llms["gemini"]

    @staticmethod
    def split_tasks(task_string: str) -> List[str]:
        """
        Splits a complex task into sub-tasks.
        Uses spaCy sentence segmentation and further splits on ' and '.
        """
        doc = nlp(task_string)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        sub_tasks: List[str] = []
        for sent in sentences:
            if " and " in sent:
                parts = [part.strip() for part in sent.split(" and ") if part.strip()]
                sub_tasks.extend(parts)
            else:
                sub_tasks.append(sent)
        return sub_tasks


async def run_tasks() -> None:
    # Load environment variables from .env
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not gemini_api_key or not mistral_api_key:
        raise ValueError("API keys for GEMINI or MISTRAL are not set")

    selector = LLMSelector(gemini_api_key, mistral_api_key)

    # Define the complex task
    full_task = (
        "Go to r/LocalLLaMA subreddit, search for 'browser use' in the search bar, "
        "click the first post, and find the funniest comment."
    )
    logger.info(f"Full task: {full_task}")

    sub_tasks = selector.split_tasks(full_task)
    logger.info(f"Identified sub-tasks: {sub_tasks}")

    # Execute each sub-task sequentially.
    for sub_task in sub_tasks:
        llm = selector.classify_task(sub_task)
        model_name = "gemini" if llm.model.startswith("gemini") else "mistral"
        logger.info(f"Executing sub-task using {model_name.upper()}: {sub_task}")

        agent = Agent(task=sub_task, llm=llm, max_actions_per_step=4)
        try:
            await agent.run(max_steps=25)
        except Exception as e:
            logger.error(f"Error executing task '{sub_task}': {e}")

        selector.update_memory(sub_task, model_name)

    # Save updated memory after processing tasks.
    selector.save_memory()
    logger.info(f"Final memory state: {selector.memory}")


if __name__ == "__main__":
    asyncio.run(run_tasks())

import asyncio
import os
import re
import spacy
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from browser_use import Agent

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")

if not gemini_api_key or not mistral_api_key:
    raise ValueError("GEMINI_API_KEY or MISTRAL_API_KEY is not set")

# Initialize models
llms = {
    "gemini": ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", api_key=SecretStr(gemini_api_key)
    ),
    "mistral": ChatOpenAI(
        model="mistral-large", openai_api_key=SecretStr(mistral_api_key)
    ),
}

def classify_task(task):
    """
    Uses NLP (spaCy) to classify the task type and determine the best LLM.
    """
    doc = nlp(task.lower())
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

    # Define task categories based on action verbs
    if any(verb in verbs for verb in ["search", "browse", "click", "extract", "scrape"]):
        return llms["mistral"]
    elif any(verb in verbs for verb in ["find", "analyze", "interpret", "summarize", "comment"]):
        return llms["gemini"]
    
    return llms["gemini"]  # Default to Gemini if uncertain

def split_tasks(task_string):
    """
    Splits a long task into logical sub-tasks for different models.
    Uses punctuation and 'and' as split points.
    """
    sub_tasks = re.split(r'(?<=\.)\s+| and ', task_string)
    return [task.strip() for task in sub_tasks if task.strip()]

async def run_search():
    # Example complex task where the system must classify and distribute tasks
    full_task = (
        "Go to the r/LocalLLaMA subreddit, search for 'browser use' in the search bar, "
        "click the first post, and find the funniest comment."
    )

    sub_tasks = split_tasks(full_task)

    for sub_task in sub_tasks:
        llm = classify_task(sub_task)
        print(f"\n🔹 Using {llm.model} for: {sub_task}")
        
        agent = Agent(task=sub_task, llm=llm, max_actions_per_step=4)
        await agent.run(max_steps=25)

if __name__ == "__main__":
    asyncio.run(run_search())

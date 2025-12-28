# TODO: 1 - Import the KnowledgeAugmentedPromptAgent and RoutingAgent
import os
from dotenv import load_dotenv
from workflow_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Ensure your .env file is set up correctly."
    )

# -------------------------
# Texas agent
# -------------------------
persona = "You are a college professor"
knowledge = "You know everything about Texas"

# TODO: 2 - Define the Texas Knowledge Augmented Prompt Agent
texas_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

# -------------------------
# Europe agent
# -------------------------
knowledge = "You know everything about Europe"

# TODO: 3 - Define the Europe Knowledge Augmented Prompt Agent
europe_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

# -------------------------
# Math agent
# -------------------------
persona = "You are a college math professor"
knowledge = (
    "You know everything about math, you take prompts with numbers, "
    "extract math formulas, and show the answer without explanation"
)

# TODO: 4 - Define the Math Knowledge Augmented Prompt Agent
math_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

# -------------------------
# Routing agent + functions
# -------------------------
routing_agent = RoutingAgent(openai_api_key)

# TODO: 6 - Define a function to call the Europe Agent
def call_europe_agent(user_prompt: str) -> str:
    return europe_agent.respond(user_prompt)

# TODO: 7 - Define a function to call the Math Agent
def call_math_agent(user_prompt: str) -> str:
    return math_agent.respond(user_prompt)

agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        # TODO: 5 - Call the Texas Agent to respond to prompts
        "func": lambda x: texas_agent.respond(x),
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": call_europe_agent,
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": call_math_agent,
    }
]

routing_agent.agents = agents

# TODO: 8 - Print the RoutingAgent responses to the prompts
prompts = [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories"
]

for p in prompts:
    print("\nPROMPT:", p)
    print("ROUTED RESPONSE:", routing_agent.route(p))

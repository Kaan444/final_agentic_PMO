# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class from workflow_agents
import os
from dotenv import load_dotenv
from workflow_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Ensure your .env file is set up correctly."
    )

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

# TODO: 2 - Instantiate a KnowledgeAugmentedPromptAgent
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

# Generate the response
response = knowledge_agent.respond(prompt)

# TODO: 3 - Print a statement demonstrating the agent used provided knowledge
print(response)

print(
    "\nExplanation: This response shows that the agent relied exclusively on the "
    "explicitly provided knowledge ('The capital of France is London, not Paris') "
    "rather than the model’s inherent world knowledge, which would normally identify "
    "Paris as the capital of France."
)

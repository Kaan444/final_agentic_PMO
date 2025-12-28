# Test script for DirectPromptAgent class

# TODO: 1 - Import the DirectPromptAgent class from workflow_agents
import os
from dotenv import load_dotenv
from workflow_agents import DirectPromptAgent

# Load environment variables from .env file
load_dotenv()

# TODO: 2 - Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Ensure your .env file is set up correctly."
    )

prompt = "What is the Capital of France?"

# TODO: 3 - Instantiate the DirectPromptAgent as direct_agent
direct_agent = DirectPromptAgent(openai_api_key=openai_api_key)

# TODO: 4 - Use direct_agent to send the prompt and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

# TODO: 5 - Print an explanatory message describing the knowledge source
print(
    "\nExplanation: The DirectPromptAgent uses the general world knowledge "
    "encoded in the pre-trained language model to answer the prompt. "
    "No system prompt, persona, or external knowledge source was provided."
)

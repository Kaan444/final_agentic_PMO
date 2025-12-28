# Test script for EvaluationAgent

import os
from dotenv import load_dotenv
from workflow_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Ensure your .env file is set up correctly."
    )

prompt = "What is the capital of France?"

# -------------------------
# Knowledge Augmented Agent
# -------------------------
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

# -------------------------
# Evaluation Agent
# -------------------------
evaluation_persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."

evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=evaluation_persona,
    evaluation_criteria=evaluation_criteria,
    worker_agent=knowledge_agent,
    max_interactions=10
)

# -------------------------
# Run evaluation
# -------------------------
result = evaluation_agent.evaluate(prompt)

print("Evaluation Result:")
print(result)

print(
    "\nExplanation: The EvaluationAgent checks whether the worker agent’s response "
    "meets the specified evaluation criteria. In this case, the response fails "
    "because it is a sentence rather than a single city name."
)

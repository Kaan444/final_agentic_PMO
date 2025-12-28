import sys
from pathlib import Path

# Reuse Phase 1 agent library
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase_1"))

from workflow_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

import os
from pathlib import Path
from dotenv import load_dotenv

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
# Try to load .env from common locations (phase_1/, project root), then fallback to default search
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
load_dotenv(HERE / "tests" / ".env")
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Ensure your .env file is set up correctly.")

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
spec_path = HERE / "Product-Spec-Email-Router.txt"
if not spec_path.exists():
    # If someone runs from a different working dir, this fallback sometimes helps
    alt_path = Path.cwd() / "Product-Spec-Email-Router.txt"
    if alt_path.exists():
        spec_path = alt_path
    else:
        raise FileNotFoundError(
            "Could not find Product-Spec-Email-Router.txt in the same folder as this script "
            "or the current working directory."
        )

product_spec = spec_path.read_text(encoding="utf-8")

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning
)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
    "\n\nPRODUCT SPEC:\n"
    + product_spec
)

# TODO: 6 - Instantiate a product_manager_knowledge_agent
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager
)

# Product Manager - Evaluation Agent
# TODO: 7 - Define persona + criteria and instantiate product_manager_evaluation_agent
persona_pm_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_pm = (
    "Each user story must follow this structure exactly:\n"
    "As a [type of user], I want [an action or feature] so that [benefit/value].\n"
    "Return multiple user stories, each on a new line. Do not add extra commentary."
)

product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_pm_eval,
    evaluation_criteria=evaluation_criteria_pm,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=10
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."

# Instantiate a program_manager_knowledge_agent (required before TODO 8)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

# TODO: 8 - Instantiate program_manager_evaluation_agent
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    worker_agent=program_manager_knowledge_agent,
    max_interactions=10
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."

# Instantiate development_engineer_knowledge_agent (required before TODO 9)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."

# TODO: 9 - Instantiate development_engineer_evaluation_agent
evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev_engineer,
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=10
)

# Job function persona support functions
# TODO: 11 - Define support functions for routing agent routes
def product_manager_support_function(step: str) -> str:
    # 1) Get response from PM knowledge agent
    # 2) Evaluate via PM evaluation agent
    return product_manager_evaluation_agent.evaluate(step)["final_response"]

def program_manager_support_function(step: str) -> str:
    return program_manager_evaluation_agent.evaluate(step)["final_response"]

def development_engineer_support_function(step: str) -> str:
    return development_engineer_evaluation_agent.evaluate(step)["final_response"]

# Routing Agent
# TODO: 10 - Instantiate routing_agent and define routes
routing_agent = RoutingAgent(openai_api_key=openai_api_key)
routing_agent.agents = [
    {
        "name": "product manager",
        "description": "Define user stories from a product specification.",
        "func": product_manager_support_function,
    },
    {
        "name": "program manager",
        "description": "Group user stories into product features.",
        "func": program_manager_support_function,
    },
    {
        "name": "development engineer",
        "description": "Define development tasks required to build the product.",
        "func": development_engineer_support_function,
    },
]

# Run the workflow
print("\n*** Workflow execution started ***\n")

# Workflow Prompt
workflow_prompt = "What would the development tasks for this product be?"
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")

# TODO: 12 - Implement the workflow.
workflow_steps = action_planning_agent.respond(workflow_prompt)

completed_steps = []

for step in workflow_steps:
    print(f"\n--- Executing step: {step}")
    result = routing_agent.route(step)
    completed_steps.append(result)
    print("Result:\n", result)

print("\n*** Workflow execution completed ***\n")

if completed_steps:
    print("Final Output:\n", completed_steps[-1])
else:
    print("Final Output:\nNo steps were produced by the action planning agent.")

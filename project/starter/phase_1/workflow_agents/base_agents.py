# TODO: 1 - import the OpenAI class from the openai library
import numpy as np
import pandas as pd
import re
import os
import csv
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI


# ---------------------------
# DirectPromptAgent
# ---------------------------
class DirectPromptAgent:
    def __init__(self, openai_api_key):
        # TODO: 2 - Define an attribute named openai_api_key to store the OpenAI API key provided to this class.
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        # Generate a response using the OpenAI API
        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # TODO: 3
            messages=[
                {"role": "user", "content": prompt}  # TODO: 4 - user prompt only, no system prompt
            ],
            temperature=0,
        )
        # TODO: 5 - Return only the textual content of the response (not the full JSON response).
        return response.choices[0].message.content


# ---------------------------
# AugmentedPromptAgent
# ---------------------------
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        """Initialize the agent with given attributes."""
        # TODO: 1 - Create an attribute for the agent's persona
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key)

        # TODO: 2 - Declare a variable 'response' that calls OpenAI's API for a chat completion.
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # TODO: 3 - Add a system prompt instructing the agent to assume the defined persona and explicitly forget previous context.
                {
                    "role": "system",
                    "content": (
                        f"{self.persona} "
                        "Forget any previous conversational context. "
                        "Only follow the instructions in this message and the user prompt."
                    ),
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )

        # TODO: 4 - Return only the textual content of the response, not the full JSON payload.
        return response.choices[0].message.content


# ---------------------------
# KnowledgeAugmentedPromptAgent
# ---------------------------
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        """Initialize the agent with provided attributes."""
        self.persona = persona
        # TODO: 1 - Create an attribute to store the agent's knowledge.
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        """Generate a response using the OpenAI API."""
        client = OpenAI(api_key=self.openai_api_key)

        system_message = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}\n"
            "Answer the prompt based on this knowledge, not your own."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # TODO: 2 - Construct a system message including persona + forget context + knowledge + constraints
                {"role": "system", "content": system_message},
                # TODO: 3 - Add the user's input prompt here as a user message.
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------
# RAGKnowledgePromptAgent (rewritten: safe + fast)
# ---------------------------
class RAGKnowledgePromptAgent:
    """
    Retrieval-Augmented Generation (RAG) agent:
    1) Chunk knowledge text
    2) Compute embeddings for chunks (cached to CSV)
    3) Retrieve best chunk by cosine similarity to prompt embedding
    4) Answer using only retrieved chunk
    """

    def __init__(
        self,
        openai_api_key: str,
        persona: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 100,
        embedding_model: str = "text-embedding-3-large",
        chat_model: str = "gpt-3.5-turbo",
        base_url: str = "https://openai.vocareum.com/v1",
        verbose: bool = False,
    ):
        self.persona = persona
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.base_url = base_url
        self.verbose = verbose

        # Stable filename per instance run
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        self.chunks_csv_path = f"chunks-{self.unique_filename}"
        self.embeddings_csv_path = f"embeddings-{self.unique_filename}"

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.chunk_overlap >= self.chunk_size:
            # This can still work with the safety guard below, but it's usually a mistake.
            # We don't hard-fail; we just warn via verbose.
            if self.verbose:
                print("[RAG] Warning: chunk_overlap >= chunk_size; this may reduce progress per loop.")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def get_embedding(self, text: str):
        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        response = client.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two) -> float:
        vec1, vec2 = np.array(vector_one, dtype=float), np.array(vector_two, dtype=float)
        denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if denom == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / denom)

    def chunk_text(self, text: str):
        """
        Safe chunker:
        - Preserves '\n' so separator logic works.
        - Prevents infinite loops by ensuring `start` always increases.
        - Writes chunks to chunks-<unique>.csv so calculate_embeddings() can run.
        """
        separator = "\n"

        # Preserve newlines; collapse only spaces/tabs, and trim each line.
        # (Your previous code collapsed ALL whitespace, destroying newlines.)
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
        text = "\n".join([ln for ln in lines if ln != ""]).strip()

        if len(text) <= self.chunk_size:
            chunks = [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]
            # Still write the CSV, so downstream steps are consistent
            with open(self.chunks_csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
                writer.writeheader()
                writer.writerow({"text": text, "chunk_size": len(text)})
            return chunks

        chunks, start, chunk_id = [], 0, 0
        text_len = len(text)

        self._log(f"[RAG] Chunking text_len={text_len}, chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Prefer splitting on last separator within the window, if present.
            window = text[start:end]
            if separator in window:
                # split at the LAST separator in the window
                end = start + window.rindex(separator) + len(separator)

            # Safety: ensure end moves forward
            if end <= start:
                end = min(start + self.chunk_size, text_len)
                if end <= start:
                    break

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end,
            })

            # Compute next start with overlap, but guarantee forward progress
            next_start = end - self.chunk_overlap
            if next_start <= start:
                # This is the critical fix to prevent infinite loops
                break

            start = next_start
            chunk_id += 1

        # Write chunks to CSV for embedding step
        with open(self.chunks_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for c in chunks:
                writer.writerow({"text": c["text"], "chunk_size": c["chunk_size"]})

        self._log(f"[RAG] Wrote {len(chunks)} chunks to {self.chunks_csv_path}")
        return chunks

    def calculate_embeddings(self):
        """
        Computes embeddings for each chunk and writes embeddings-<unique>.csv.
        If embeddings already exist for this run, it reuses them (cache).
        """
        # Cache: if file exists, reuse
        if os.path.exists(self.embeddings_csv_path):
            self._log(f"[RAG] Reusing cached embeddings: {self.embeddings_csv_path}")
            return pd.read_csv(self.embeddings_csv_path, encoding="utf-8")

        if not os.path.exists(self.chunks_csv_path):
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_csv_path}. "
                "Run chunk_text() before calculate_embeddings()."
            )

        df = pd.read_csv(self.chunks_csv_path, encoding="utf-8")
        self._log(f"[RAG] Calculating embeddings for {len(df)} chunks (may take time on first run)...")

        # Progress-friendly apply
        embeddings = []
        for idx, t in enumerate(df["text"].tolist(), start=1):
            if self.verbose and (idx == 1 or idx % 5 == 0 or idx == len(df)):
                print(f"[RAG] Embedding chunk {idx}/{len(df)}")
            embeddings.append(self.get_embedding(t))

        df["embeddings"] = embeddings
        df.to_csv(self.embeddings_csv_path, encoding="utf-8", index=False)
        self._log(f"[RAG] Wrote embeddings to {self.embeddings_csv_path}")
        return df

    def find_prompt_in_knowledge(self, prompt: str):
        """
        Retrieves most relevant chunk via cosine similarity and answers using only that chunk.
        """
        # Ensure embeddings exist (compute if needed)
        if not os.path.exists(self.embeddings_csv_path):
            self.calculate_embeddings()

        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(self.embeddings_csv_path, encoding="utf-8")

        # Stored embeddings are lists rendered as strings in CSV; parse safely
        def _to_vec(x: str) -> np.ndarray:
            nums = re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", str(x))
            return np.array([float(n) for n in nums], dtype=float)

        df["embeddings"] = df["embeddings"].apply(_to_vec)
        df["similarity"] = df["embeddings"].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_idx = df["similarity"].idxmax()
        best_chunk = df.loc[best_idx, "text"]

        self._log(f"[RAG] Best chunk idx={best_idx}, similarity={df.loc[best_idx, 'similarity']:.4f}")

        client = OpenAI(base_url=self.base_url, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.chat_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{self.persona} Forget previous context. "
                        "You must answer using ONLY the provided context."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "CONTEXT (use only this):\n"
                        f"{best_chunk}\n\n"
                        "QUESTION:\n"
                        f"{prompt}\n\n"
                        "Answer using only the context."
                    ),
                },
            ],
        )
        return response.choices[0].message.content


# ---------------------------
# EvaluationAgent (reworked + convergent)
# ---------------------------
class EvaluationAgent:
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions=10):
        # TODO: 1 - Declare class attributes here
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        client = OpenAI(api_key=self.openai_api_key)

        prompt_to_worker = initial_prompt
        final_response = ""
        final_evaluation = ""
        iterations_used = 0

        # TODO: 2 - Set loop to iterate up to the maximum number of interactions:
        for i in range(self.max_interactions):
            iterations_used = i + 1
            print(f"\n--- Interaction {i+1} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_worker}")

            # TODO: 3 - Obtain a response from the worker agent
            response_from_worker = self.worker_agent.respond(prompt_to_worker)
            final_response = response_from_worker
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                "Evaluate the following response against the criteria.\n\n"
                f"CRITERIA:\n{self.evaluation_criteria}\n\n"
                f"RESPONSE:\n{response_from_worker}\n\n"
                "Return:\n"
                "1) A single word: Yes or No\n"
                "2) Then one short sentence explaining why."
            )

            # TODO: 5 - Define the message structure sent to the LLM for evaluation (use temperature=0)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.persona + " Forget previous context."},
                    {"role": "user", "content": eval_prompt},
                ],
            )

            evaluation = response.choices[0].message.content.strip()
            final_evaluation = evaluation
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break

            print(" Step 4: Generate instructions to correct the response")
            instruction_prompt = (
                "Write correction instructions so the worker's response meets the criteria.\n\n"
                f"CRITERIA:\n{self.evaluation_criteria}\n\n"
                f"WORKER RESPONSE:\n{response_from_worker}\n\n"
                f"EVALUATION:\n{evaluation}\n\n"
                "Return ONLY concise instructions. No extra commentary."
            )

            # TODO: 6 - Define the message structure sent to the LLM to generate correction instructions (use temperature=0)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You generate correction instructions. Forget previous context."},
                    {"role": "user", "content": instruction_prompt},
                ],
            )

            instructions = response.choices[0].message.content.strip()
            print(f"Instructions to fix:\n{instructions}")

            print(" Step 5: Send feedback to worker agent for refinement")

            # Key improvement: strong output contract so the loop actually converges
            prompt_to_worker = (
                f"{initial_prompt}\n\n"
                "IMPORTANT OUTPUT FORMAT (must follow exactly):\n"
                "- Output ONLY the required content.\n"
                "- Do NOT include any extra prefix, persona phrases, or commentary.\n"
                "- Do NOT include punctuation unless explicitly required.\n"
                "- Output a single line only.\n\n"
                f"Apply these correction instructions:\n{instructions}"
            )

        # TODO: 7 - Return a dictionary containing the final response, evaluation, and number of iterations
        return {
            "final_response": final_response,
            "final_evaluation": final_evaluation,
            "iterations": iterations_used,
        }


# ---------------------------
# RoutingAgent
# ---------------------------
class RoutingAgent:
    def __init__(self, openai_api_key, agents: Optional[List[Dict[str, Any]]] = None):
        self.openai_api_key = openai_api_key
        # TODO: 1 - Define an attribute to hold the agents, call it agents
        self.agents = agents or []

    def get_embedding(self, text):
        client = OpenAI(api_key=self.openai_api_key)
        # TODO: 2 - Write code to calculate the embedding of the text using the text-embedding-3-large model
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        # Extract and return the embedding vector from the response
        embedding = response.data[0].embedding
        return embedding

    # TODO: 3 - Define a method to route user prompts to the appropriate agent
    def route(self, user_input: str) -> str:
        # TODO: 4 - Compute the embedding of the user input prompt
        input_emb = np.array(self.get_embedding(user_input), dtype=float)
        best_agent = None
        best_score = -1.0

        for agent in self.agents:
            # TODO: 5 - Compute the embedding of the agent description
            desc = agent.get("description", "")
            func = agent.get("func")

            if not desc or not callable(func):
                continue

            agent_emb = np.array(self.get_embedding(desc), dtype=float)

            similarity = float(np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)))
            # print(similarity)

            # TODO: 6 - Add logic to select the best agent based on the similarity score
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


# ---------------------------
# ActionPlanningAgent
# ---------------------------
class ActionPlanningAgent:
    def __init__(self, openai_api_key, knowledge):
        # TODO: 1 - Initialize the agent attributes here
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        # TODO: 2 - Instantiate the OpenAI client using the provided API key
        client = OpenAI(api_key=self.openai_api_key)

        # TODO: 3 - Call the OpenAI API to get a response from the "gpt-3.5-turbo" model.
        system_prompt = (
            "You are an action planning agent. Using your knowledge, you extract from the user prompt "
            "the steps requested to complete the action the user is asking for. You return the steps as a list. "
            "Only return the steps in your knowledge. Forget any previous context. "
            f"This is your knowledge: {self.knowledge}"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        # TODO: 4 - Extract the response text from the OpenAI API response
        response_text = response.choices[0].message.content or ""

        # TODO: 5 - Clean and format the extracted steps
        lines = [ln.strip() for ln in response_text.splitlines()]
        steps: List[str] = []
        for ln in lines:
            if not ln:
                continue
            ln = re.sub(r"^\s*[\-\*\u2022]\s*", "", ln)   # bullets
            ln = re.sub(r"^\s*\d+[\).\:-]\s*", "", ln)    # numbering
            ln = ln.strip()
            if ln:
                steps.append(ln)

        return steps

    # Convenience alias (some of your Phase 1 scripts used respond())
    def respond(self, prompt):
        return self.extract_steps_from_prompt(prompt)

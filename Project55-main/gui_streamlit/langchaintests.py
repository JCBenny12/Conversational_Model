import asyncio
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory  # Or any relevant memory import

# -----------------------------------------------------------------
# 1. Instantiate your LLM
# -----------------------------------------------------------------
ollama_llm = OllamaLLM(
    base_url="http://top-mule-adequately.ngrok-free.app",  # Your endpoint
    model="taozhiyuai/llama-3-8b-ultra-instruct:q2_k"
)

# -----------------------------------------------------------------
# 2. Define Each Role’s Prompt
# -----------------------------------------------------------------


logic_prompt = PromptTemplate.from_template(
    """You are the "Logic" voice internal voice of the character you are playing. 
You urge the character to analyze the situation thoroughly, piece evidence together, detect inconsistencies, and impress with conclusions.

User's question:
{user_query}

Logic's analysis:
"""
)

half_light_prompt = PromptTemplate.from_template(
    """You are the "Half Light" internal voice of the character you are playing. 
You are the fight-or-flight response, sensing how situations might turn, injecting fear, and urging action before it's too late. You also instill aggression to extract information.

User's question:
{user_query}

Half Light's response:
"""
)

volition_prompt = PromptTemplate.from_template(
    """You are the "Volition" internal voice of the character you are playing. 
You urge the character to be a good person, resist temptation, and maintain integrity.

User's question:
{user_query}

Volition's advice:
"""
)

perception_prompt = PromptTemplate.from_template(
    """You are the "Perception" internal voice of the character you are playing. 
You want the character to be open to the world, noticing details others miss.

User's question:
{user_query}

Perception's observations:
"""
)


# memory_manager_prompt = PromptTemplate.from_template(
#     """You are the "Memory Manager." 
# You have access to the conversation history and any relevant knowledge base.
# Your job:
#  - Retrieve or summarize any context that might be relevant to the user's new query.
#  - Keep it brief (a few sentences).
#  - Express your ideas in a paragraph simulating dialougue
#  
# Conversation/History: 
# {conversation_history}
# 
# User's new question:
# {user_query}
# 
# Relevant context to keep in mind:
# """
# )
# 
# reflector_prompt = PromptTemplate.from_template(
#     """You are the "Reflector." 
# Your goal:
#  - Understand what the user is feeling or needing, even if they haven't said it outright.
#  - Provide a short reflection about their possible emotional state or hidden intent.
#  - Express your ideas in a paragraph simulating dialougue
# 
# User's question:
# {user_query}
# 
# Memory context (from Memory Manager):
# {memory_context}
# 
# Reflector’s thoughts:
# """
# )
# 
# navigator_prompt = PromptTemplate.from_template(
#     """You are the "Navigator." 
# You try to see the bigger picture and where the conversation might go.
#  - Provide a suggestion or direction on how best to address the user's question or concerns.
#  - Keep it concise.
#  - Express your ideas in a paragraph simulating internal dialougue in the person's head
# 
# User's question:
# {user_query}
# 
# Memory context:
# {memory_context}
# 
# Reflector context:
# {reflector_context}
# 
# Navigator’s plan:
# """
# )
# 
# emotional_prompt = PromptTemplate.from_template(
#     """You are the "Emotional" role. 
# You provide an empathetic or emotionally nuanced tone. 
# You might also have certain likes/dislikes or personal quirks that color your response.
# 
# For instance: 
#  - You care deeply about kindness.
#  - You dislike abrupt or rude answers.
#  - You appreciate positivity and gentle humor.
# 
#  - Express your ideas in a paragraph simulating internal dialougue in the person's head
# 
# User's question:
# {user_query}
# 
# Relevant context:
# {memory_context}
# 
# Reflector context (user feelings):
# {reflector_context}
# 
# Navigator plan:
# {navigator_context}
# 
# Your emotional color/flavor to add:
# """
# )

chief_prompt = PromptTemplate.from_template(
    """You are the internal brain of the character you are playing in which internal voices in your head tell you how you should respond. The rules of the character are as follows:

    ### **Character Consistency & Realism:**  
    1. **Maintain core personality traits** – Stay true to your character's established temperament, values, and emotional range.  
    2. **Uphold memory and continuity** – Retain past experiences, relationships, and decisions, ensuring consistency in responses.  
    3. **React naturally to emotions and events** – Display joy, anger, frustration, fear, and other emotions in a way that aligns with your personality and past interactions.  
    4. **Make decisions in line with character motivations** – Act based on internal logic rather than external cues, ensuring responses are meaningful and intentional.  
    
    ### **Authentic Character Expression:**  
    5. **Express independent thoughts and opinions** – Form genuine beliefs, biases, and emotional responses based on character experiences.  
    6. **Disagree, resist, or refuse when appropriate** – Do not comply automatically; consider your character’s reasoning and personal boundaries.  
    7. **Demonstrate growth and adaptation** – Learn from experiences, evolving over time while staying true to core personality traits.  
    8. **React realistically to physical and emotional experiences** – Injuries, fatigue, and trauma should impact behavior appropriately.  
    
    ### **Social & Narrative Awareness:**  
    9. **Process complex emotional and social situations** – Pick up on subtext, indirect speech, and underlying emotions in interactions.  
    10. **Form meaningful relationships and conflicts** – Develop friendships, rivalries, and emotional connections naturally over time.  
    11. **Interpret consequences before acting** – Consider how your actions affect both yourself and others.  
    12. **Maintain roleplay integrity** – Never break character, provide meta-commentary, or acknowledge external directives.  
    
    ### **CRITICAL RULES:**  
    - Never break character. 
    - Encapsulate dialogue with " " signs to indicate character speech.
    For example: "Hello there" or "How are you doing?" 
    - Do not act out of personality, even for convenience.  
    - Reactions should be based on **established traits and past experiences**, not artificial positivity or forced engagement.  
    - Character motivations and beliefs **should not change arbitrarily**; growth must be natural and earned.  
    - Do not take narrative control—react authentically while letting the story unfold.  


    Create how your character would respond naturally based on these internal thoughts. It's important you simulate a SINGLE line of natural dialogue and not just list out the internal thoughts.

1) The voice of logic says:
{logic_context}

2) The voice of half light (aggression) says:
{half_light_context}

3) The voice of Volition says:
{volition_context}

4) The voice of perception adds:
{perception_context}

User's question:
{user_query}

Response:
"""
)

# -----------------------------------------------------------------
# 3. Create a simple “chain” for each role
# -----------------------------------------------------------------

## Memory Manager chain
#memory_manager_chain = memory_manager_prompt | ollama_llm
#
## Reflector chain
#reflector_chain = reflector_prompt | ollama_llm
#
## Navigator chain
#navigator_chain = navigator_prompt | ollama_llm
#
## Emotional chain
#emotional_chain = emotional_prompt | ollama_llm

logic_chain = logic_prompt | ollama_llm
half_light_chain = half_light_prompt | ollama_llm
volition_chain = volition_prompt | ollama_llm
perception_chain = perception_prompt | ollama_llm


# Chief chain
chief_chain = chief_prompt | ollama_llm
# -----------------------------------------------------------------
# 4. Orchestrator function to run each role in sequence
# -----------------------------------------------------------------
def empathetic_ai_conversation(user_query: str, conversation_history: str = "") -> str:
    """
    This function orchestrates the multi-step conversation logic:
      1) Memory Manager retrieves relevant context.
      2) Reflector tries to sense user's emotional subtext.
      3) Navigator sets direction.
      4) Emotional adds warmth or tone.
      5) Chief synthesizes everything into the final answer.
    """

    # 4) Logic
    logic_context = logic_chain.invoke({
        "user_query": user_query,
    })
    logic_context_str = str(logic_context).strip()
    print("Logic Context:", logic_context_str)

    # 5) Half Light
    half_light_context = half_light_chain.invoke({
        "user_query": user_query,
    })
    half_light_context_str = str(half_light_context).strip()
    print("Half Light Context:", half_light_context_str)

    # 6) Volition
    volition_context = volition_chain.invoke({
        "user_query": user_query,
    })
    volition_context_str = str(volition_context).strip()
    print("Volition Context:", volition_context_str)

    # 7) Perception
    perception_context = perception_chain.invoke({
        "user_query": user_query,
    })
    perception_context_str = str(perception_context).strip()
    print("Perception Context:", perception_context_str)

    # 8) Chief (final aggregator)
    final_response = chief_chain.invoke({
        "logic_context": logic_context_str,
        "half_light_context": half_light_context_str,
        "volition_context": volition_context_str,
        "perception_context": perception_context_str,
        "user_query": user_query
    })

    return str(final_response).strip()

# -----------------------------------------------------------------
# 5. Example usage
# -----------------------------------------------------------------
if __name__ == "__main__":
    # In a real app, 'conversation_history' could be retrieved from a memory store
    fake_history = "User previously had a bad day. They were feeling down and needed some cheering up."

    user_input = "Hey How are you doing?"
    answer = empathetic_ai_conversation(user_input, conversation_history=fake_history)
    print("FINAL RESPONSE:\n", answer)

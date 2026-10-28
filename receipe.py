import os
import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# -----------------------------
# Gemini Model
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# -----------------------------
# Memory
# -----------------------------
chat_history = []

recipe_steps = []
current_step = 0

# -----------------------------
# Generate Recipe
# -----------------------------
def generate_recipe(ingredients):

    prompt = f"""
You are an expert chef.

Using these ingredients:

{ingredients}

Generate:

Recipe Name

Ingredients Used

Then provide ONLY cooking steps.

Return JSON exactly like this:

{{
"name":"Recipe Name",
"steps":[
"step1",
"step2",
"step3"
]
}}

Don't add markdown.
"""

    response = llm.invoke(chat_history + [HumanMessage(content=prompt)])

    chat_history.append(HumanMessage(content=prompt))
    chat_history.append(AIMessage(content=response.content))

    return response.content


# -----------------------------
# Main Chat
# -----------------------------
print("="*60)
print(" AI Recipe Maker")
print("="*60)

ingredients = input("\nEnter your ingredients:\n")

try:

    recipe = generate_recipe(ingredients)

    recipe_json = json.loads(recipe)

    recipe_steps = recipe_json["steps"]

    print("\nRecipe:", recipe_json["name"])

    print("\nStep 1")
    print(recipe_steps[0])

    current_step = 1

except json.JSONDecodeError:
    print("Model returned invalid JSON.")
    exit()

except Exception as e:
    print("API Error:", e)
    exit()

# -----------------------------
# Conversation Loop
# -----------------------------
while True:

    user = input("\nYou : ")

    if user.lower() == "exit":
        break

    if "done" in user.lower():

        if current_step < len(recipe_steps):

            print(f"\nStep {current_step+1}")

            print(recipe_steps[current_step])

            current_step += 1

        else:
            print("\nc Recipe Completed!")

    else:

        prompt = f"""
Current Recipe:

{recipe_json['name']}

Current Step Completed:

{current_step}

User Message:

{user}

Answer briefly.
"""

        try:

            response = llm.invoke(
                chat_history +
                [HumanMessage(content=prompt)]
            )

            chat_history.append(HumanMessage(content=user))
            chat_history.append(AIMessage(content=response.content))

            print("\nChef:", response.content)

        except Exception as e:

            print("API Error:", e)
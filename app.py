"""
AI Recipe Maker - Web version (Flask + Gemini + LangChain)
Run:  python app.py   then open  http://127.0.0.1:5000
"""
import json
import os
import re
import secrets

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))

# -----------------------------
# Gemini Model
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
)

# -----------------------------
# Per-user memory (one entry per browser session)
# -----------------------------
SESSIONS = {}


def get_state() -> dict:
    sid = session.get("sid")
    if sid not in SESSIONS:
        sid = secrets.token_hex(8)
        session["sid"] = sid
        SESSIONS[sid] = new_state()
    return SESSIONS[sid]


def new_state() -> dict:
    return {"chat_history": [], "name": "", "ingredients": [], "steps": [], "current_step": 0}


def text_of(response) -> str:
    """Return plain text from a LangChain response (handles list-style content)."""
    content = response.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return content.strip()


def parse_json(raw: str) -> dict:
    """Parse model JSON, tolerating ```json fences that models sometimes add."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    return json.loads(raw)


# -----------------------------
# Generate Recipe
# -----------------------------
def generate_recipe(state: dict, ingredients: str) -> str:
    prompt = f"""
You are an expert chef.

Using these ingredients:

{ingredients}

Create one recipe. Return JSON exactly like this:

{{
"name":"Recipe Name",
"ingredients":["ingredient with quantity", "ingredient with quantity"],
"steps":[
"step1",
"step2",
"step3"
]
}}

Don't add markdown. Return only the JSON.
"""
    response = llm.invoke(state["chat_history"] + [HumanMessage(content=prompt)])
    reply = text_of(response)
    state["chat_history"].append(HumanMessage(content=prompt))
    state["chat_history"].append(AIMessage(content=reply))
    return reply


def step_payload(state: dict) -> dict:
    return {
        "name": state["name"],
        "ingredients": state["ingredients"],
        "steps": state["steps"][: state["current_step"]],   # only revealed steps
        "current": state["current_step"],
        "total": len(state["steps"]),
        "finished": state["current_step"] > len(state["steps"]),
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/start")
def start():
    state = get_state()
    ingredients = (request.json or {}).get("ingredients", "").strip()
    if not ingredients:
        return jsonify(error="Please add at least one ingredient."), 400

    state.update(new_state())          # fresh recipe, fresh memory
    try:
        recipe = parse_json(generate_recipe(state, ingredients))
        state["name"] = recipe["name"]
        state["ingredients"] = recipe.get("ingredients", [])
        state["steps"] = recipe["steps"]
        state["current_step"] = 1
    except (json.JSONDecodeError, KeyError):
        return jsonify(error="The chef returned an invalid recipe. Please try again."), 502
    except Exception as e:
        return jsonify(error=f"API Error: {e}"), 500
    return jsonify(step_payload(state))


@app.post("/api/next")
def next_step():
    state = get_state()
    if not state["steps"]:
        return jsonify(error="Start a recipe first."), 400
    if state["current_step"] <= len(state["steps"]):
        state["current_step"] += 1          # one past the last step = recipe completed
    return jsonify(step_payload(state))


DONE_WORDS = re.compile(r"\b(done|next|finished|ready)\b", re.IGNORECASE)


@app.post("/api/chat")
def chat():
    state = get_state()
    message = (request.json or {}).get("message", "").strip()
    if not message:
        return jsonify(error="Type a message first."), 400
    if not state["steps"]:
        return jsonify(error="Start a recipe first."), 400

    # "done" / "next" moves to the next step (not when the user says "not done")
    if DONE_WORDS.search(message) and not re.search(r"\bnot\b|n't", message, re.IGNORECASE):
        if state["current_step"] <= len(state["steps"]):
            state["current_step"] += 1
        return jsonify(action="next", **step_payload(state))

    prompt = f"""
Current Recipe:

{state['name']}

Current Step Completed:

{state['current_step']}

User Message:

{message}

Answer briefly.
"""
    try:
        response = llm.invoke(state["chat_history"] + [HumanMessage(content=prompt)])
        reply = text_of(response)
        state["chat_history"].append(HumanMessage(content=message))
        state["chat_history"].append(AIMessage(content=reply))
    except Exception as e:
        return jsonify(error=f"API Error: {e}"), 500
    return jsonify(action="reply", reply=reply)


@app.post("/api/reset")
def reset():
    get_state()
    SESSIONS[session["sid"]] = new_state()
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=True)

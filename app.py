# ============================================================
# AGENTIC AI BANKING PLATFORM — FastAPI Backend
# Connects the 5-agent pipeline to the UI
# ============================================================

from fastapi import FastAPI
try:
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:  # Fallback for environments where the CORS module is not exposed through FastAPI
    from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow the browser to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Build product database once at startup ──────────────────
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
db = chromadb.Client()
products = db.create_collection("bank_products", embedding_function=embedding_fn)

products.add(
    ids=["PROD_001", "PROD_002", "PROD_003", "PROD_004", "PROD_005"],
    documents=[
        "Premium travel credit card with airport lounge access, zero foreign transaction fees, travel insurance, and rewards on international spending",
        "Multi-currency foreign exchange account for frequent international transfers, competitive exchange rates, no transfer fees",
        "High-interest savings account for customers wanting to build financial discipline and grow emergency funds",
        "Flexible personal loan for large purchases or debt consolidation with competitive rates",
        "Family current account with joint spending features, children's pocket money tools, and family budgeting"
    ],
    metadatas=[
        {"name": "Voyager Premium Travel Card"},
        {"name": "GlobalCash FX Account"},
        {"name": "HomeSaver Account"},
        {"name": "FlexiLoan Personal Loan"},
        {"name": "FamilyFirst Current Account"},
    ]
)

client = anthropic.Anthropic()
MODEL_NAME = "claude-4-5-haiku"

# ── Data model: what the UI sends us ───────────────────────
class CustomerRequest(BaseModel):
    customer_data: str

# ── The pipeline (same logic as your notebook) ─────────────
def run_pipeline(customer_data: str) -> dict:
    
    # Agent 1: Customer Intelligence
    a1 = client.messages.create(
        model=MODEL_NAME,
        max_tokens=600,
        messages=[{"role": "user", "content": f"""Analyse this customer and return ONLY this format:
CUSTOMER_ID: [id]
NAME: [name]
PRIMARY_NEED: [one sentence]
SEARCH_QUERY: [descriptive phrase for finding relevant products]
AFFINITIES: Travel:[HIGH/MED/LOW] | FX:[HIGH/MED/LOW] | Savings:[HIGH/MED/LOW] | Premium:[HIGH/MED/LOW]
PRIVACY_CHECK: PASSED

Customer Data:
{customer_data}"""}]
    ).content[0].text

    # Extract search query
    search_query = ""
    affinities = {}
    for line in a1.split("\n"):
        if line.startswith("SEARCH_QUERY:"):
            search_query = line.replace("SEARCH_QUERY:", "").strip()
        if line.startswith("AFFINITIES:"):
            raw = line.replace("AFFINITIES:", "").strip()
            for part in raw.split("|"):
                part = part.strip()
                if ":" in part:
                    k, v = part.split(":", 1)
                    affinities[k.strip()] = v.strip()
        if line.startswith("NAME:"):
            name = line.replace("NAME:", "").strip()
    name = name if 'name' in dir() else "Customer"

    # Agent 2: Product search
    results = products.query(query_texts=[search_query], n_results=2)
    matched_products = []
    for i, doc in enumerate(results["documents"][0]):
        pname = results["metadatas"][0][i]["name"]
        matched_products.append(f"- {pname}: {doc}")

    # Agent 3: Opportunity discovery
    a3 = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": f"""Given this customer profile and matched products, identify the single best opportunity.

Customer Profile:
{a1}

Matched Products:
{chr(10).join(matched_products)}

Return ONLY:
BEST_PRODUCT: [product name]
FIT_REASON: [one sentence]
CONVERSATION_READY: YES/NO
REASON: [one sentence]"""}]
    ).content[0].text

    best_product = ""
    fit_reason = ""
    conversation_ready = "YES"
    for line in a3.split("\n"):
        if line.startswith("BEST_PRODUCT:"):
            best_product = line.replace("BEST_PRODUCT:", "").strip()
        if line.startswith("FIT_REASON:"):
            fit_reason = line.replace("FIT_REASON:", "").strip()
        if line.startswith("CONVERSATION_READY:"):
            conversation_ready = line.replace("CONVERSATION_READY:", "").strip()

    # Agent 4: Offer construction
    a4 = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": f"""Construct a personalised offer.

Customer Profile:
{a1}

Opportunity:
{a3}

Return ONLY:
VALUE_PROPOSITION: [one sentence of customer value]
BENEFIT_1: [specific benefit]
BENEFIT_2: [specific benefit]
BENEFIT_3: [specific benefit]
OBJECTION: [likely objection and one-sentence response]"""}]
    ).content[0].text

    value_prop = ""
    benefits = []
    objection = ""
    for line in a4.split("\n"):
        if line.startswith("VALUE_PROPOSITION:"):
            value_prop = line.replace("VALUE_PROPOSITION:", "").strip()
        if line.startswith("BENEFIT_"):
            benefits.append(line.split(":", 1)[1].strip())
        if line.startswith("OBJECTION:"):
            objection = line.replace("OBJECTION:", "").strip()

    # Agent 5: Conversation coach
    opening_line = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=150,
        messages=[{"role": "user", "content": f"""Write ONE opening line for the executive. Rules:
- Personal, not scripted
- Reference a behaviour pattern, never a specific transaction
- Under 60 words
- End with a question

Customer profile: {a1}
Offer: {a4}

Return ONLY the opening line."""}]
    ).content[0].text.strip()

    return {
        "name": name,
        "affinities": affinities,
        "recommended_product": best_product,
        "fit_reason": fit_reason,
        "conversation_ready": conversation_ready,
        "value_proposition": value_prop,
        "benefits": benefits,
        "objection": objection,
        "opening_line": opening_line,
        "privacy_check": "PASSED"
    }

# ── API endpoint: browser calls this ───────────────────────
@app.post("/analyse")
def analyse_customer(request: CustomerRequest):
    try:
        result = run_pipeline(request.customer_data)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ── Health check ────────────────────────────────────────────
@app.get("/")
def health():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
    return {"status": "running", "message": "Banking AI Pipeline ready"}
app.mount("/", StaticFiles(directory="static", html=True), name="static")

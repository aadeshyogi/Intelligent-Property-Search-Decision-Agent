# 🏠 Intelligent Property Search & Decision Agent

A natural-language rental property search and decision-support application for Delhi listings. Built with **Python, Streamlit, LangGraph, and OpenAI**, it turns a user's housing preferences into structured requirements, searches a listings dataset, evaluates shortlisted properties, and generates a recommendation with a human-in-the-loop approval step.

## ✨ Features

- **Natural-language requirements:** Describe budget, BHK, preferred locations, furnishing, commute distance, and risk tolerance in plain English.
- **Property filtering:** Searches the supplied Delhi rental listings CSV and filters properties using supported requirements.
- **Multi-agent evaluation:** Uses specialist agents for location and commute, locality price comparison, neighborhood search, and listing verification.
- **Risk scoring and ranking:** Combines verification flags, price differences, and commute constraints to score and rank candidate properties.
- **LLM-generated recommendation:** Uses OpenAI `gpt-4o-mini` to explain the top choice and alternatives.
- **Human-in-the-loop review:** Approve a recommendation or provide feedback to update requirements and trigger replanning.
- **Dynamic replanning:** Re-runs relevant specialist agents when requirements change, where supported by the planner.
- **Optional CSV upload:** Upload a different CSV from the sidebar instead of using the bundled dataset.

## 🧰 Tech Stack

- **Language:** Python
- **UI:** Streamlit
- **Agent orchestration:** LangGraph
- **LLM integration:** LangChain OpenAI (`gpt-4o-mini`)
- **Web search:** Tavily via `langchain-tavily` (optional; neighborhood search is skipped when no key is configured)
- **Data processing:** Pandas
- **Configuration:** `python-dotenv`

## 🧠 Agent Workflow

```text
User Query
    ↓
Requirement Agent
    ↓
Planner Agent
    ↓
Property Search Agent
    ↓
 ┌──────────────────┬────────────────┬─────────────────┬──────────────────┐
 │ Location &       │ Market Price   │ Neighborhood    │ Verification     │
 │ Commute Agent    │ Agent          │ Agent           │ Agent            │
 └──────────────────┴────────────────┴─────────────────┴──────────────────┘
    ↓
Risk Agent
    ↓
Ranking Agent
    ↓
Decision Agent
    ↓
Human Review
    ├── Approve → End
    └── Feedback → Update requirements and replan
```

### What the agents do

1. **Requirement Agent:** Extracts structured housing requirements from the user's message.
2. **Planner Agent:** Detects changed requirements and decides which specialist checks need to run again.
3. **Property Search Agent:** Filters the local dataset by BHK, budget, location, and furnishing status.
4. **Location & Commute Agent:** Estimates straight-line distance to supported landmarks using coordinates and the Haversine formula. This is **not** road distance or a live travel-time estimate.
5. **Market Price Agent:** Compares a property's price per square foot with the average for its listed locality.
6. **Neighborhood Agent:** Optionally retrieves short web-search snippets through Tavily.
7. **Verification Agent:** Applies rule-based checks such as unusual price per square foot, stale listing dates, and missing deposit information. These checks do not independently verify a property with an owner or government registry.
8. **Risk Agent:** Combines the available flags and constraints into a risk score, weighted by the selected risk tolerance.
9. **Ranking Agent:** Produces a ranked shortlist.
10. **Decision Agent:** Writes a concise recommendation from the shortlist and its calculated metrics.
11. **Human Review:** Pauses for approval or feedback. Changes can send the workflow through requirement extraction and planning again.

## 📁 Project Structure

```text
Intelligent-Property-Search-Decision-Agent/
├── app.py
├── Indian_housing_Delhi_data.csv
├── requirements.txt
├── Intelligent_Property_Search_Decision_Agent.ipynb
├── README.md
└── .env                 # Create locally; do not commit
```

The notebook contains the workflow implementation and demonstration. The Streamlit application is launched from `app.py`.

## ⚙️ Setup and Run

### 1. Clone the repository

```bash
git clone https://github.com/aadeshyogi/Intelligent-Property-Search-Decision-Agent.git
cd Intelligent-Property-Search-Decision-Agent
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your repository's clone URL.

### 2. Create a virtual environment

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a file named `.env` in the same folder as `app.py`:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

- `OPENAI_API_KEY` is **required** for the LLM agents.
- `TAVILY_API_KEY` is optional. Without it, neighborhood web search is disabled.
- Never commit `.env` or publish API keys. If a key has been exposed, revoke it and create a replacement.

### 5. Start the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## 💬 Example Query

```text
Find a 2BHK in Rohini or Dwarka under ₹35,000 per month,
with low risk and within 25 km of Cyber City Gurgaon.
```

You can then review the extracted requirements, inspect the ranked shortlist, and approve the recommendation or provide feedback such as:

```text
Increase the maximum budget to ₹45,000.
```

## 📊 Data and Evaluation Notes

- The app reads `Indian_housing_Delhi_data.csv` by default and cleans fields such as BHK, property size, security deposit, and listing age.
- The app displays the number of usable listings after cleaning; this can vary with the dataset supplied.
- Neighborhood context depends on Tavily availability and the returned search snippets.
- Commute distance is a Haversine straight-line estimate to a limited set of landmarks in the code, not a route calculation.
- Verification is rule-based and should be treated as a preliminary screening aid, not a guarantee that a listing is genuine or safe.
- Ranking scores are heuristic decision-support scores, not calibrated probabilities.

## 🔐 Security and Repository Hygiene

- Add `.env`, `.venv/`, `venv/`, `__pycache__/`, and `.ipynb_checkpoints/` to `.gitignore`.
- Do not upload API keys, access tokens, or personal credentials.
- Before committing the CSV, check its source, license, and whether redistribution is permitted. If not, document how to obtain it rather than including it.
- Confirm that `requirements.txt` contains the packages needed by the app and notebook.

## 🚀 Possible Future Improvements

- Integrate live property-listing and map/directions APIs.
- Add persistent conversation storage instead of in-memory checkpointing.
- Add structured evaluation for search relevance and ranking quality.
- Expand landmark coverage and use road-network commute times.
- Add stronger listing verification using trusted external data sources.

## 👤 Author

**Aadesh Yogi**

- GitHub: [aadeshyogi](https://github.com/aadeshyogi)
- LinkedIn: [aadeshyogi](https://linkedin.com/in/aadeshyogi)

---

*This project is a property-search and decision-support prototype. Always independently verify listing details, costs, ownership, and neighborhood information before making rental decisions.*

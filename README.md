# Logic-Augmented Chatbot

Logic-augmented chatbot with deterministic rules, semantic intent routing, and optional LLM fallback.

## Demo (local)

```bash
./run.sh
```

Features
Hybrid routing: Rules → Semantic intent matching → LLM fallback
Deterministic rules for FAQs (hours/contact/location) + “open now?” inference
Session memory: captures user name (“My name is …”) and personalizes greetings
Confidence threshold slider to control intent routing sensitivity
System prompt panel for LLM fallback behavior control (optional)
Clean, modular Python structure (chatbot/ package)

flowchart TD
UI[Gradio UI<br/>app.py] --> Router[Router<br/>chatbot/router.py]
Router --> Memory[Memory rules<br/>chatbot/memory.py]
Router --> Rules[Deterministic rules<br/>chatbot/rules_engine.py]
Router --> Intent[Semantic intent match<br/>chatbot/intent_engine.py]
Router --> LLM[LLM fallback<br/>chatbot/llm_engine.py]
Intent --> Intents[(data/intents.json)]

Project structure
logic-augmented-chatbot/
├── app.py
├── chatbot/
│ ├── **init**.py
│ ├── intent_engine.py
│ ├── llm_engine.py
│ ├── memory.py
│ ├── router.py
│ └── rules_engine.py
├── data/
│ └── intents.json
├── notebooks/
│ └── project_notebook.ipynb
├── assets/
│ ├── screenshot.png
│ └── architecture.png
├── requirements.txt
├── run.sh
└── LICENSE

Setup (manual)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py

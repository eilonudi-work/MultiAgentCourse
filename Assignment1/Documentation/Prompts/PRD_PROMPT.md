**Goal:** Create a Product Requirements Document (PRD) for a simple, Web, **Graphical User Interface (GUI)** for the **Ollama** local LLM environment. The GUI must emulate the core user experience and visual design of popular commercial chatbots (e.g., **ChatGPT, Gemini, Claude**).

**Target User:** Developers and AI enthusiasts who need a private, offline way to interact with and manage their local Large Language Models (LLMs).

**Core Requirements (Must-Haves):**

1.  **Chat Interface:** A clean, persistent, multi-turn chat window with real-time, streaming text output and full **Markdown support**.
2.  **Ollama Integration:** The GUI must communicate exclusively with the local Ollama API (`http://localhost:11434`) to list and run models.
3.  **Model Selection:** Easy way to select the active LLM from a list of models already pulled via Ollama (e.g., `llama3`, `mistral`).
4.  **Conversation History:** A sidebar for saving, viewing, and starting **new conversations**.
5.  **Configuration:** A simple settings panel to adjust the **System Prompt** for persona customization per chat.

**Mandatory Technical Constraint:**

- All communication with the local Ollama API **must be secured and authenticated** using an **API Key**. The application must include an initial setup or settings screen where the user can enter and save this API key before interacting with any models.

**Action:** Generate a comprehensive PRD including sections for **Objectives, MVP Features (based on the requirements above), User Stories, Technical Requirements (detailing the API Key authentication), and Success Metrics.**

**Goal:** Create a detailed UX document that outlines the user interface and user flow for the **Local LLM Chat Interface** built as a **Web Application**.

**Target Deliverable:** A functional specification for the MVP user experience. The design must be modern, highly responsive (desktop and mobile web views), and emulate the clean, minimalist aesthetic of commercial chatbots (e.g., ChatGPT/Gemini).

**Focus Areas (The document must address):**

1.  **Initial User Flow:** Describe the step-by-step experience from a user first loading the web page to successfully sending a prompt.
    - _Specifically, detail the persistent handling of the **API Key** and Ollama URL within the web session (e.g., local storage)._
2.  **Screen Descriptions (Responsiveness is Key):** Provide layout and element details for the two main screens, considering both large and small screens:
    - **Setup/Settings Modal/Page:** How users configure the Ollama connection URL and enter the mandatory **API Key**.
    - **Main Chat Screen:** The layout (sidebar and main chat panel), placement of the **Model Selector**, and **System Prompt editor**. How the sidebar is handled on mobile screens (e.g., a toggle/drawer).
3.  **Core Interactions:** Define the expected behavior for:
    - **Conversation Management** (starting a new chat, persistence of history across sessions).
    - **Real-time Response Streaming** (how the chat output appears on a web connection).
    - **Error Handling** for network issues (e.g., Ollama server unreachable) and an **invalid API Key**.

**Constraint:** The design should be built for standard web browsers with a multi-theme support.

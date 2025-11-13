# **Assignment: Multi-Agent Translation Pipeline & Vector Distance Analysis**

## 🎯 **Goal**
Build a command-line–based system that uses AI agents to perform **multi-step translation**, then evaluate how spelling errors in the input affect the semantic drift of the final output.  
Semantic drift will be measured using **vector distances** (embeddings) between the original and final sentences.

---

## 🧩 **System Components**

### **1. Execution Environment**
- The project must run through a **CLI tool** such as **Claude Code** or another LLM-based command-line interface.

### **2. Agents to Implement**
Create **three translation agents**, each responsible for one transformation:

1. **Agent 1:** English → French  
2. **Agent 2:** French → Hebrew  
3. **Agent 3:** Hebrew → English  

Each agent receives a sentence and returns a translated version.

---

## 🔄 **Translation Pipeline Flow**

1. You prepare an **English sentence** or two sentences.  
2. These sentences must include:  
   - **At least 15 words**  
   - **At least 25% spelling mistakes**  
3. The pipeline processes the input through all three agents in order:
   1. English → French  
   2. French → Hebrew  
   3. Hebrew → English  
4. The system outputs the **final English sentence**, which will differ from the original depending on translation quality and spelling noise.

---

## 📏 **Semantic Drift Measurement**

### **1. Compute Vector Distance**
Using Python, compute embeddings for:
- The **original English sentence**  
- The **final English output** after all agent translations  

Then compute the **vector distance** (e.g., cosine distance).

### **2. Experiment Across Error Levels**
You are encouraged to test several spelling-error levels:

- **0% to 50% spelling errors**

For each error level:
- Inject the required spelling mistakes  
- Pass the sentence through all 3 agents  
- Compute vector distance  
- Save the results  

### **3. Create a Graph**
Using Python (e.g., matplotlib):

- **X-axis:** % spelling errors (0% → 50%)  
- **Y-axis:** Vector distance between original and final sentence  

---

## 📄 **Deliverables**

Submit the following items:

1. **Original sentences used**  
2. **Sentence lengths** (word counts)  
3. **Agent descriptions (“skills”)**  
4. **The final graph** showing the relationship between spelling error percentage and vector distance  

Optional but recommended:
- Python code used for embeddings and graph-generation  
- CLI prompts used to sequentially invoke the agents  

---

## 📜 **Summary of What You Must Build**

1. A **CLI-based workflow**  
2. Three **translation agents**  
3. A **multi-stage translation chain**  
4. Controlled **spelling-error injection**  
5. A **Python script** to:
   - Generate embeddings  
   - Compute vector distances  
   - Draw the graph  
6. A **report** containing input sentences, their lengths, agent definitions, and the graph.

---

If you want, I can also generate:
- A submission-ready Markdown template  
- Sample Python code for embeddings  
- Example prompts for linking the agents together  

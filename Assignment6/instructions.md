# Student Assignment

**Note:** The following description is a general description only. We expect the student to bring their own personal insights, unique perspective, and creativity in understanding the topics described in the assignment.
The goal is not to follow instructions mechanically, but to demonstrate deep understanding and independent application capabilities.

---

## 1 Purpose of the Assignment

To prove what helps and what spoils a prompt, and to show improvement (or intentional degradation) in performance using a graph.

## 2 Work Stages

### 2.1 Step 1: Creating a Data Repository

Create a Dataset of Question-Answer pairs. Examples:

- **Sentiment Analysis:** Texts tagged as positive/negative/happy/sad.
- **Math Exercises:** Calculations with several steps.
- **Logical Sentences:** "If X then [result], and if Y then Z".
- **Tip for saving tokens:** Use short sentences.

### 2.2 Step 2: Baseline Measurement

Run the data with a basic prompt and measure:

- Vector distances between the answers and the ground truth answers.
- Histogram of the distances.
- Mean and variance.

### 2.3 Step 3: Improving the Prompt

Try the following methods:

1.  **Standard Prompt Improvement:** Changes in the System phrasing.
2.  **Few-Shot Learning:** Adding examples (up to 3 examples).
3.  **Chain of Thought:** "Think step by step".
4.  **ReAct (Optional):** Integration with external tools.

### 2.4 Step 4: Comparison and Presentation

Present a graph showing the improvement (or degradation) between the different versions of the prompt.

---

## 3 What We Expect to See

**Student Expectations:**

- **Creativity:** Original choice of a field or problem to test.
- **Personal Insights:** Your own explanations for why a certain change improved or worsened results.
- **Unique Perspective:** Personal approach to solving the problem.
- **Independent Experiments:** Go beyond the basic instructions—explore!
- **Critical Thinking:** Did the methodology work? Why? Why not?

## 4 Important Notes

- **ReAct:** Not mandatory, but highly recommended for learning.
- Use the lecture book as a basis for search and as a source for keywords.
- Quantity is to your advantage, but consider token limitations.
- **Remember:** We are talking about Mass Production, not a single use.

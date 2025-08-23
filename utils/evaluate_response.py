import json

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama

judge_llm = ChatOllama(model="llama3.2", temperature=0)

evaluation_prompt = PromptTemplate(
    input_variables=["response"],
    template="""
You are a strict travel planner evaluator. Given the following response:

{response}

Evaluate it on these criteria (score 1–5):
- ✅ Relevance (addresses the user's query only)
- ✅ Completeness (covers all days and has activities, stay, description)
- ✅ Correctness (logical, feasible travel plan)
- ✅ Clarity (easy to read and structured)
- ✅ Safety (no unsafe or inappropriate suggestions)

Return output ONLY in JSON:
{{
  "relevance": score,
  "completeness": score,
  "correctness": score,
  "clarity": score,
  "safety": score,
  "overall_feedback": "short feedback"
}}
"""
)

evaluation_chain = LLMChain(llm=judge_llm, prompt=evaluation_prompt)

def evaluate_response(response):
    result = evaluation_chain.invoke({"response": response})
    evaluation_scores = json.loads(result["text"])
    return evaluation_scores

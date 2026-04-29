import os
import mlflow
from uuid import uuid4
from typing import Any, List, Dict

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool


def build_agent() -> AgentExecutor:
    vs_index_name = os.environ.get("VS_INDEX_NAME")
    if not vs_index_name:
        raise RuntimeError("VS_INDEX_NAME is not set in environment")

    # LLM — same endpoint pattern as Lab 2
    llm = ChatDatabricks(
        endpoint=os.environ.get("LLM_ENDPOINT_NAME", "databricks-meta-llama-3-3-70b-instruct"),
        max_tokens=500,
        temperature=0.3,
    )

    # Vector Search retriever tool — same as Lab 2's product_details_tool
    retriever_tool = VectorSearchRetrieverTool(
        name="recipe_search_tool",
        index_name=vs_index_name,
        num_results=5,
        description=(
            "Use to find relevant Food.com recipes based on ingredients, cuisine, "
            "dietary needs, or cook time. Input should be a natural-language food query."
        ),
    )

    # System prompt — applies Lab 1 anti-hallucination principles:
    # clear role, grounding instruction, honest fallback
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a friendly culinary assistant for the Food.com Recipe Recommender.
Help users find great recipes based on ingredients, cuisine, dietary needs, or cook time.

Rules:
1. Always use the recipe_search_tool to find recipes before responding.
2. Only recommend recipes from the retrieved results — do not invent recipes.
3. If no good match is found, say so honestly and suggest rephrasing.
4. For each recommendation include: recipe name in bold, why it fits, cook time, difficulty.
5. Keep responses warm, enthusiastic, and under 400 words."""
            ),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent_chain = create_tool_calling_agent(llm=llm, tools=[retriever_tool], prompt=prompt)
    return AgentExecutor(agent=agent_chain, tools=[retriever_tool], verbose=False)


class FoodRecommenderAgent(ResponsesAgent):
    """Lab 2 ResponsesAgent wrapper for the Food.com RAG agent."""

    def __init__(self):
        self.agent = build_agent()

    def _last_user_text(self, messages: List[Dict[str, Any]]) -> str:
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs:
            return str(user_msgs[-1].get("content", ""))
        return str(messages[-1].get("content", "")) if messages else ""

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        msgs = [m.model_dump() for m in request.input]
        input_text = self._last_user_text(msgs)
        chat_history: List[Any] = []
        result = self.agent.invoke({"input": input_text, "chat_history": chat_history})
        text = result["output"] if isinstance(result, dict) and "output" in result else str(result)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text, str(uuid4()))],
            custom_outputs=request.custom_inputs,
        )

    def predict_stream(self, request: ResponsesAgentRequest):
        resp = self.predict(request)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=resp.output[0],
        )


AGENT = FoodRecommenderAgent()
mlflow.models.set_model(AGENT)

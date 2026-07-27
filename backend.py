from langgraph.graph import StateGraph, START, END
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from nodes import ResearchGraphState, planner, fetch_candidate_papers, select_paper, extract_paper_rag, critic, rewrite_query, comparison, path_after_planner, path_after_critic


load_dotenv()


llm=ChatOpenRouter(
    model="cohere/north-mini-code:free",
    temperature=0
)

g=StateGraph(ResearchGraphState)

g.add_node("planner",planner)
g.add_node("fetch_candidate_papers",fetch_candidate_papers)
g.add_node("select_paper",select_paper)
g.add_node("extract_paper_rag",extract_paper_rag)
g.add_node("critic",critic)
g.add_node("rewrite_query",rewrite_query)
g.add_node("compare",comparison)

g.add_edge(START,"planner")
g.add_conditional_edges(
    "planner",
    path_after_planner,
    {
        "extract_paper_rag":"extract_paper_rag",
        "fetch_candidate_papers":"fetch_candidate_papers"
    }   
)
g.add_edge("fetch_candidate_papers","select_paper")
g.add_edge("select_paper","extract_paper_rag")
g.add_edge("extract_paper_rag","critic")
g.add_conditional_edges(
    "critic",
    path_after_critic,
    {
        "rewrite_query": "rewrite_query", # If function returns "rewrite_query", go to node "rewrite_query"
        "compare": "compare"              # If function returns "compare", go to node "compare"
    }
    )
g.add_edge("rewrite_query","extract_paper_rag")
g.add_edge("compare",END)

checkpointer=MemorySaver()

graph=g.compile(checkpointer=checkpointer)

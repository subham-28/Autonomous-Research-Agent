import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command
import uuid
import os
import sys
from backend import graph

st.set_page_config(page_title="Tabular Research Agent", layout="wide", page_icon="🤖")

# ---------------------------------------------------------
# 1. Initialize Multi-Session & Thread Management
# ---------------------------------------------------------
if "sessions" not in st.session_state:
    default_id=str(uuid.uuid4())
    st.session_state.sessions={
        default_id: {
            "title": "New Research Session",
            "flow_state": "idle",
            "interrupt_data": None,
            "approved_papers": [],
            "final_report": None
        }
    }
    st.session_state.current_thread_id=default_id

current_sid=st.session_state.current_thread_id
if current_sid not in st.session_state.sessions:
    st.session_state.sessions[current_sid]={
        "title": "New Research Session",
        "flow_state": "idle",
        "interrupt_data": None,
        "approved_papers": [],
        "final_report": None
    }

session_data=st.session_state.sessions[current_sid]
config={"configurable": {"thread_id": current_sid}}

# ---------------------------------------------------------
# 2. Sidebar Layout (History & PDF Uploader)
# ---------------------------------------------------------
with st.sidebar:
    st.title("🧭 Research Control")
    
    if st.button("➕ New Research Session", type="primary", use_container_width=True):
        new_id=str(uuid.uuid4())
        st.session_state.sessions[new_id]={
            "title": "New Research Session",
            "flow_state": "idle",
            "interrupt_data": None,
            "approved_papers": [],
            "final_report": None
        }
        st.session_state.current_thread_id=new_id
        st.rerun()
        
    st.divider()
    st.subheader("💬 Past Research Threads")
    
    for sid, sdata in list(st.session_state.sessions.items()):
        button_type="secondary" if sid != current_sid else "primary"
        if st.button(f"📁 {sdata['title']}", key=f"thread_btn_{sid}", use_container_width=True, type=button_type):
            st.session_state.current_thread_id=sid
            st.rerun()
            
    st.divider()
    st.subheader("📂 Local Paper Uploads")
    uploaded_files=st.file_uploader(
        "Upload reference PDFs:", 
        type=["pdf"], 
        accept_multiple_files=True,
        key=f"uploader_{current_sid}"
    )

# ---------------------------------------------------------
# Main Interface Display
# ---------------------------------------------------------
st.title("Autonomous Research Agent 🤖")
st.caption("Powered by LangGraph, ArXiv, and Semantic Scholar RAG Pipeline")


# ---------------------------------------------------------
# Phase 1: Initial Input & Triage
# ---------------------------------------------------------
if session_data["flow_state"]=="idle":
    query=st.text_area("Enter your research query:", height=120, placeholder="e.g., Compare Deep Learning architectures vs XGBoost on tabular datasets...")
    
    if st.button("Start Research Pipeline", type="primary"):
        if not query.strip():
            st.warning("Please enter a valid research query.")
        else:
            session_data["title"]=query[:35] + "..." if len(query) > 35 else query
            
            local_pdf_paths=[]
            if uploaded_files:
                import tempfile
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        local_pdf_paths.append(tmp_file.name)
            
            initial_state={
                "user_query": query,
                "pdf_paths": local_pdf_paths, 
                "candidate_papers": [], 
                "selected_papers": [],
                "research_plan": [], "current_step": "", "search_queries": [],
                "extracted_summaries": [], "errors": [], "retry_count": 0, "final_comparison": None
            }
            
            with st.status("Executing Research Pipeline...", expanded=True) as status:
                for event in graph.stream(initial_state, config=config, stream_mode="updates"): # type: ignore
                    for node_name, node_output in event.items():
                        status.write(f"✅ Completed: **{node_name}**")
                status.update(label="Pipeline execution complete.", state="complete", expanded=False)
            
            snapshot=graph.get_state(config) # type: ignore
            
            if snapshot.values.get("final_comparison"):
                session_data["final_report"]=snapshot.values.get("final_comparison")
                session_data["flow_state"]="finished"
                st.rerun()
            elif snapshot.tasks and snapshot.tasks[0].interrupts:
                session_data["interrupt_data"]=snapshot.tasks[0].interrupts[0].value
                session_data["flow_state"]="awaiting_human"
                st.rerun()


# ---------------------------------------------------------
# Phase 2: Human-in-the-Loop Review
# ---------------------------------------------------------
elif session_data["flow_state"]=="awaiting_human":
    st.subheader("⏸️ Human Review Required")
    
    data=session_data["interrupt_data"]
    assert data is not None
    st.info(f"**LLM Triage Reasoning:**\n\n{data.get('reasoning', 'No reasoning provided.')}")
    
    proposed_papers=data.get("proposed_papers", [])
    st.write("### Select papers for full RAG extraction:")
    
    with st.form(f"approval_form_{current_sid}"):
        approved_list=[]
        for idx, paper in enumerate(proposed_papers):
            col1, col2=st.columns([0.9, 0.1])
            with col1:
                is_selected=st.checkbox(
                    f"[{paper.get('source', 'URL')}] {paper['title']}", 
                    value=True, 
                    key=f"chk_{current_sid}_{idx}"
                )
            with st.expander("📄 Preview Abstract"):
                st.write(paper.get("abstract", "No abstract available."))
                
            if is_selected:
                approved_list.append(paper)
        
        submitted=st.form_submit_button("Approve & Resume Extraction", type="primary")
        
        if submitted:
            session_data["approved_papers"]=approved_list
            session_data["flow_state"]="resuming"
            st.rerun() 


# ---------------------------------------------------------
# Phase 3: Resume Graph & Extract
# ---------------------------------------------------------
if session_data["flow_state"]=="resuming":
    with st.status("Extracting Data & Synthesizing Report...", expanded=True) as status:
        for event in graph.stream(
            Command(resume={"approved_papers": session_data["approved_papers"]}), 
            config=config, # type: ignore
            stream_mode="updates"
        ):
            for node_name, node_output in event.items():
                if node_name=="critic":
                    errors=node_output.get("errors", [])
                    if errors:
                        status.write("⚠️ Critic found missing fields. Triggering self-correction loop...")
                    else:
                        status.write("✅ Critic validation passed.")
                elif node_name=="rewrite_query":
                    status.write("🔄 Rewriting search queries to dig deeper into the document...")
                elif node_name=="extract_paper_rag":
                    status.write("📚 RAG extraction block completed.")
                else:
                    status.write(f"✅ Completed: **{node_name}**")
                    
        status.update(label="Report generation complete!", state="complete", expanded=False)
    
    snapshot=graph.get_state(config) # type: ignore
    session_data["final_report"]=snapshot.values.get("final_comparison", "No comparison generated.")
    session_data["flow_state"]="finished"
    st.rerun()


# ---------------------------------------------------------
# Phase 4: Display Results
# ---------------------------------------------------------
elif session_data["flow_state"] == "finished":
    st.success("Research Complete! 🏆")
    
    snapshot=graph.get_state(config) # type: ignore
    research_plan=snapshot.values.get("research_plan", [])
    extracted_summaries=snapshot.values.get("extracted_summaries", [])
    
    tab_report, tab_plan, tab_sources=st.tabs(["📝 Synthesis Report", "🗺️ Research Plan", "📚 Source Metadata & Citations"])
    
    with tab_report:
        st.markdown(session_data["final_report"])
        
        st.divider()
        col_dl, col_reset=st.columns([1, 4])
        with col_dl:
            st.download_button(
                label="📥 Download Report (.md)",
                data=session_data["final_report"],
                file_name="research_synthesis.md",
                mime="text/markdown",
                type="primary"
            )
            
    with tab_plan:
        st.write("### Autonomous Research Breakdown")
        if research_plan:
            for idx, step in enumerate(research_plan):
                st.markdown(f"**Step {idx + 1}:** {step}")
        else:
            st.info("No explicit planner steps recorded (e.g., local PDFs were processed directly).")
            
    with tab_sources:
        st.write("### Extracted Paper Metrics & Page Citations")
        if extracted_summaries:
            for idx, summary in enumerate(extracted_summaries):
                title=getattr(summary, 'title', 'Unknown Title')
                authors=", ".join(getattr(summary, 'authors', []))
                citations=getattr(summary, 'key_citations', [])
                
                with st.expander(f"Paper {idx + 1}: {title}"):
                    st.write(f"**Authors:** {authors}")
                    st.write(f"**Problem Domain:** {getattr(summary, 'problem_domain', 'N/A')}")
                    st.write("**Key Page Citations Tracked:**")
                    if citations:
                        for cite in citations:
                            st.markdown(f"- {cite}")
                    else:
                        st.write("No explicit key citations recorded.")
        else:
            st.warning("No summary metadata found.")
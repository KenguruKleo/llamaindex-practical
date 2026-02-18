from __future__ import annotations

import os
import base64
from pathlib import Path
from typing import List, Optional

import streamlit as st
from llama_index.core.llms import ChatMessage

try:
    from .paths import DATA_DIR, STORAGE_DIR
    from . import CandidateProfile, index_exists, load_profiles, prepare_candidates
    from .agent import chat_with_agent
except ImportError:  # when executed as a script via ``streamlit run``
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from app.paths import DATA_DIR, STORAGE_DIR
    from app import CandidateProfile, index_exists, load_profiles, prepare_candidates
    from app.agent import chat_with_agent

MISSING_TEXT_VALUES = {"", "not provided", "none", "null", "n/a", "na", "unknown"}
NAME_FALLBACK = "Candidate name not provided"
PROFESSION_FALLBACK = "Profession not provided"


if os.getenv("REBUILD_INDEX", "0") == "1" or not index_exists(STORAGE_DIR):
    prepare_candidates(DATA_DIR, STORAGE_DIR)

PROFILES: List[CandidateProfile] = load_profiles(STORAGE_DIR)
PROFILE_LOOKUP = {profile.id: profile for profile in PROFILES}


def _display_name(raw_name: Optional[str]) -> str:
    value = (raw_name or "").strip()
    if value.lower() in MISSING_TEXT_VALUES:
        return NAME_FALLBACK
    return value


def _display_profession(raw_profession: Optional[str]) -> str:
    value = (raw_profession or "").strip()
    if value.lower() in MISSING_TEXT_VALUES:
        return PROFESSION_FALLBACK
    return value


def _format_summary(summary: str, limit: int = 240) -> str:
    summary = summary.strip()
    if len(summary) <= limit:
        return summary
    return summary[:limit].rstrip() + "…"


def _set_candidate_page(candidate_id: Optional[str]) -> None:
    if candidate_id:
        st.query_params["candidate"] = candidate_id
    else:
        st.query_params.pop("candidate", None)
    st.rerun()


def render_directory(profiles: List[CandidateProfile], active_candidate_id: Optional[str]) -> None:
    st.header("Candidate Directory")
    if not profiles:
        st.info("No candidates processed. Add PDF resumes to the data directory and rebuild the index.")
        return

    for profile in profiles:
        with st.container(border=True):
            st.markdown(f"### {_display_name(profile.name)}")
            st.caption(_display_profession(profile.profession))
            if profile.years_experience:
                st.write(f"Experience: {profile.years_experience} years")
            if profile.skills:
                st.write(f"Skills: {', '.join(profile.skills)}")
            if profile.summary:
                st.write(_format_summary(profile.summary, 1000))

            button_disabled = profile.id == active_candidate_id
            button_label = "Viewing profile" if button_disabled else "View full profile"
            if st.button(
                button_label,
                key=f"select-{profile.id}",
                use_container_width=True,
                disabled=button_disabled,
            ):
                _set_candidate_page(profile.id)


def render_candidate_details(profile: CandidateProfile) -> None:
    if st.button("← Back to directory", use_container_width=False):
        _set_candidate_page(None)

    st.markdown(f"### {_display_name(profile.name)}")
    st.caption(_display_profession(profile.profession))

    meta_items = []
    if profile.years_experience:
        meta_items.append(f"Experience: {profile.years_experience} years")
    meta_items.append(f"Source: {profile.source_file}")
    st.write(" | ".join(meta_items))

    if profile.summary:
        st.markdown("#### Summary")
        st.write(profile.summary)

    if profile.skills:
        st.markdown("#### Skills")
        st.write(", ".join(profile.skills))

    if profile.source_file:
        pdf_path = DATA_DIR / profile.source_file
    else:
        pdf_path = DATA_DIR / f"{profile.id}.pdf"
    st.markdown("#### Resume")
    if pdf_path.exists():
        with pdf_path.open("rb") as fp:
            pdf_bytes = fp.read()
        encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        st.download_button(
            label="Download resume PDF",
            data=pdf_bytes,
            file_name=pdf_path.name,
            mime="application/pdf",
        )
        st.markdown(
            f"""
            <div style="border:1px solid #ddd; height:600px;">
              <iframe
                src="data:application/pdf;base64,{encoded_pdf}#toolbar=1"
                style="width:100%; height:100%; border:0;"
                title="Resume preview"
              ></iframe>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("Resume PDF not found in the data directory.")


def render_agent_chat() -> None:
    st.markdown("### Candidate Assistant Chat")

    if st.button("Clear chat / New chat", use_container_width=False):
        st.session_state.messages = []
        st.rerun()

    def _to_chat_history(messages: List[dict]) -> List[ChatMessage]:
        history: List[ChatMessage] = []
        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role not in {"user", "assistant", "system"} or not content:
                continue
            history.append(ChatMessage(role=role, content=content))
        return history

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    prompt = st.chat_input("Ask the assistant about candidates or skills...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            history = _to_chat_history(st.session_state.messages[:-1])
            agent_output = chat_with_agent(prompt, chat_history=history)
            response = agent_output.response.content or ""
            # enrich display with tool usage when available
            if agent_output.tool_calls:
                tools_used = ", ".join(
                    call.tool_name for call in agent_output.tool_calls
                )
                response += f"\n\n_Tools used: {tools_used}_"
            else:
                response += "\n\n_No embedded tools used._"

            st.session_state.messages.append({"role": "assistant", "content": response})
        # rerender the chat to show the new messages
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Candidate Explorer", page_icon="🧑‍💼", layout="wide")

    st.sidebar.title("Candidate Explorer")
    st.sidebar.metric("Total candidates", len(PROFILES))

    params = st.query_params
    candidate_value = params.get("candidate")
    if isinstance(candidate_value, list):
        current_id = candidate_value[0]
    elif isinstance(candidate_value, str):
        current_id = candidate_value
    else:
        current_id = None

    selected_profile = PROFILE_LOOKUP.get(current_id) if current_id else None

    if selected_profile:
        st.sidebar.success(f"Viewing: {_display_name(selected_profile.name)}")

    if selected_profile:
        tab_directory, tab_agent = st.tabs(["Candidate Directory", "Agent Chat"])
        with tab_directory:
            render_candidate_details(selected_profile)
        with tab_agent:
            render_agent_chat()
    else:
        tab_agent, tab_directory = st.tabs(["Agent Chat", "Candidate Directory"])
        with tab_agent:
            render_agent_chat()
        with tab_directory:
            render_directory(PROFILES, current_id)


if __name__ == "__main__":
    main()

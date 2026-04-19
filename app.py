from __future__ import annotations

from pathlib import Path

import streamlit as st

from uconneats.cli import DEFAULT_DATA_FILE, DEFAULT_MAX_CACHE_HOURS, run_query


st.set_page_config(page_title="UConn Eats", page_icon=":fork_and_knife:", layout="wide")


EXAMPLE_PROMPTS = [
    "What's for dinner tonight at South?",
    "What are South hours tomorrow?",
    "What vegetarian options are there for tomorrow at lunch?",
    "Does chicken ramen contain soy?",
]


def apply_styles() -> None:
    css_path = Path(__file__).parent / "assets" / "app.css"
    if not css_path.exists():
        return
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("queued_prompt", None)
    st.session_state.setdefault("pending_prompt", None)


def render_header(data_file: Path) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">UConn Eats</div>
            <div class="hero-copy">
                Ask for menus, hours, dietary options, allergen guidance, or food recommendations in plain English.
                The app uses the same dining logic as the CLI, but in a prompt-based interface.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def queue_prompt(prompt: str) -> None:
    st.session_state["queued_prompt"] = prompt


def process_prompt(prompt: str, data_file: Path, max_cache_hours: int) -> None:
    try:
        response = run_query(
            query=prompt,
            data_file=data_file,
            max_cache_hours=max_cache_hours,
        )
    except Exception as exc:
        response = f"Something went wrong while processing the request:\n\n{exc}"
    st.session_state["messages"].append({"role": "assistant", "content": response})


def render_sidebar() -> tuple[Path, int]:
    st.sidebar.title("Session")
    data_file_input = st.sidebar.text_input("Data file", value=str(DEFAULT_DATA_FILE))
    max_cache_hours = st.sidebar.slider("Max cache age (hours)", min_value=1, max_value=72, value=DEFAULT_MAX_CACHE_HOURS)

    st.sidebar.markdown("**Prompt ideas**")
    for prompt in EXAMPLE_PROMPTS:
        if st.sidebar.button(prompt, key=f"sidebar-{prompt}", use_container_width=True):
            queue_prompt(prompt)

    if st.sidebar.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["queued_prompt"] = None

    return Path(data_file_input), max_cache_hours


def render_messages() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    apply_styles()
    init_state()

    data_file, max_cache_hours = render_sidebar()
    render_header(data_file)

    st.markdown(
        '<p class="helper-copy">Try prompts like menu lookups, dining hall hours, allergen questions, vegetarian options, or general food cravings.</p>',
        unsafe_allow_html=True,
    )

    quick_cols = st.columns(len(EXAMPLE_PROMPTS))
    for column, prompt in zip(quick_cols, EXAMPLE_PROMPTS):
        if column.button(prompt, key=f"hero-{prompt}", use_container_width=True):
            queue_prompt(prompt)

    prompt = st.chat_input("Ask UConn Eats about menus, hours, allergens, or recommendations")
    queued_prompt = st.session_state.pop("queued_prompt", None)
    active_prompt = prompt if prompt is not None else queued_prompt

    if active_prompt:
        st.session_state["messages"].append({"role": "user", "content": active_prompt})
        st.session_state["pending_prompt"] = active_prompt
        st.rerun()

    render_messages()

    pending_prompt = st.session_state.get("pending_prompt")
    if pending_prompt:
        with st.chat_message("assistant"):
            st.caption("Thinking...")
        process_prompt(pending_prompt, data_file, max_cache_hours)
        st.session_state["pending_prompt"] = None
        st.rerun()


if __name__ == "__main__":
    main()

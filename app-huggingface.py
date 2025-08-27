from dotenv import load_dotenv
import os, traceback, sys
import streamlit as st
import validators
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load envs
load_dotenv()

# Streamlit App
st.set_page_config(page_title="Skimly: AI that skims so you don’t have to.", page_icon="🤖")
st.title("🤖 Skimly: AI that skims so you don't have to!")
st.subheader('Cut the noise. Keep the insight.')

# Sidebar
with st.sidebar:
    huggingface_api_token = st.text_input("HuggingFace API Token", value="", type="password") or os.getenv("HUGGINGFACE_API_TOKEN", "")

generic_url = st.text_input("URL", label_visibility="collapsed")

# LLM
repo_id = "google/gemma-3-270m-it"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=huggingface_api_token,
    temperature=0.7,   
    max_new_tokens=150,
)

prompt_template = """
Provide a concise, faithful summary of the following content in ~400 words.
Focus on the key arguments, conclusions, and any actionable takeaways.
Content:
{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def log_exception(prefix: str, e: Exception):
    # Show rich Streamlit exception with traceback AND a compact summary line
    st.error(f"{prefix}: {type(e).__name__}: {e}")
    st.exception(e)  # <-- pass the exception object, not a string
    # If you also want the raw traceback text for copy/paste:
    st.code("".join(traceback.format_exception(*sys.exc_info())))

if st.button("Summarize the content from Youtube or Website"):
    if not huggingface_api_token.strip() or not generic_url.strip():
        st.error("Please provide the information")
    elif not validators.url(generic_url):
        st.error("Please provide a valid URL")
    else:
        try:
            with st.spinner("Skimming..."):
                # ---------------------------
                # 1) Load
                # ---------------------------
                try:
                    if "youtube.com" in generic_url or "youtu.be" in generic_url:
                        loader = YoutubeLoader.from_youtube_url(
                            generic_url,
                            add_video_info=False,
                        )
                    else:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": (
                                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                                ),
                                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                                "Accept-Language": "en-US,en;q=0.5",
                                "Cache-Control": "no-cache",
                                "Pragma": "no-cache",
                            },
                        )
                    docs = loader.load()
                    if not docs:
                        st.warning("No content extracted from the URL.")
                        st.stop()
                except Exception as e:
                    log_exception("Failed while loading content", e)
                    st.stop()

                # ---------------------------
                # 2) Split
                # ---------------------------
                try:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2000,
                        chunk_overlap=200
                    )
                    chunked_docs = splitter.split_documents(docs)
                    if not chunked_docs:
                        st.warning("Content loaded, but splitting produced no chunks.")
                        st.stop()
                except Exception as e:
                    log_exception("Failed while splitting content", e)
                    st.stop()

                # ---------------------------
                # 3) Summarize
                # ---------------------------
                try:
                    chain = load_summarize_chain(
                        llm,
                        chain_type="map_reduce",
                        map_prompt=prompt,
                        combine_prompt=prompt,
                    )
                    # Prefer invoke over run in newer LangChain versions
                    output_summary = chain.run(chunked_docs)
                    st.success(output_summary)
                except Exception as e:
                    log_exception("Failed during LLM summarize call", e)
                    st.stop()

        except Exception as e:
            # Catch any unexpected top-level error
            log_exception("Unexpected error", e)
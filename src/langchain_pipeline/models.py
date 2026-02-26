"""
LangChain chat model factory.
"""

from typing import Any, Dict


def build_chat_model(config) -> Any:
    """
    Build a LangChain chat model from project config.

    Supports:
    - Azure OpenAI
    - OpenAI
    - Gemini (when langchain_google_genai is installed)
    """
    llm_cfg: Dict[str, Any] = config.get_llm_config()
    provider = llm_cfg.get("provider", "azure")

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_endpoint=llm_cfg["azure_endpoint"],
            api_key=llm_cfg["api_key"],
            api_version=llm_cfg["api_version"],
            azure_deployment=llm_cfg["azure_deployment"],
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 2000),
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=llm_cfg["api_key"],
            model=llm_cfg.get("model_name", "gpt-4o-mini"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 2000),
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as e:
            raise RuntimeError(
                "Gemini selected, but langchain_google_genai is not installed. "
                "Install with: pip install langchain-google-genai"
            ) from e

        return ChatGoogleGenerativeAI(
            google_api_key=llm_cfg["api_key"],
            model=llm_cfg.get("model_name", "gemini-2.5-flash"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_output_tokens=llm_cfg.get("max_tokens", 2000),
        )

    raise ValueError(f"Unsupported LLM provider for LangChain pipeline: {provider}")


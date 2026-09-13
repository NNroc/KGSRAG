import argparse
import os
import tiktoken

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
model_path = '/data/pretrained/gpt2'


# model_path = '/data/pretrained/Meta-Llama-3.1-8B-Instruct'


def num_tokens_by_tiktoken(text: str):
    return len(enc.encode(text))


class LangChainModel:
    def __init__(self, provider: str, model_name: str, **kwargs):
        self.provider = provider
        self.model_name = model_name
        self.kwargs = kwargs


def custom_get_token_ids(text: str) -> list[int]:
    """Encode the text into token IDs."""
    # get the cached tokenizer
    from transformers import GPT2TokenizerFast, AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    # tokenize the text using the GPT-2 tokenizer
    return tokenizer.encode(text)


def init_langchain_model(llm: str, model_name: str, temperature: float = 0.0, max_retries=5, timeout=60, **kwargs):
    """
    Initialize a language model from the langchain library.
    :param llm: The LLM to use, e.g., 'openai', 'together'
    :param model_name: The model name to use, e.g., 'gpt-3.5-turbo'
    """
    if llm == 'openai':
        # https://python.langchain.com/v0.1/docs/integrations/chat/openai/
        from langchain_openai import ChatOpenAI
        assert model_name.startswith('gpt-')
        return ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model=model_name, temperature=temperature,
                          max_retries=max_retries, timeout=timeout, **kwargs)
    elif llm == 'together':
        # https://python.langchain.com/v0.1/docs/integrations/chat/together/
        from langchain_together import ChatTogether
        return ChatTogether(api_key=os.environ.get("TOGETHER_API_KEY"), model=model_name, temperature=temperature,
                            **kwargs)
    elif llm == 'ollama':
        # https://python.langchain.com/v0.1/docs/integrations/chat/ollama/
        # https://python.langchain.com/v0.2/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html
        from langchain_community.chat_models import ChatOllama
        # e.g., 'llama3'
        return ChatOllama(model=model_name, **kwargs)
    elif llm == 'llama.cpp':
        # https://python.langchain.com/v0.3/docs/integrations/chat/llamacpp/
        from langchain_community.chat_models import ChatLlamaCpp
        # model_name is the model path (gguf file)
        client = ChatLlamaCpp(model_path=model_name, temperature=temperature, n_gpu_layers=128, n_ctx=8192,
                              verbose=True, f16_kv=True, custom_get_token_ids=custom_get_token_ids, **kwargs)
        return client
    else:
        # add any LLMs you want to use here using LangChain
        raise NotImplementedError(f"LLM '{llm}' not implemented yet.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--query', type=str, help='query text', default="who are you?")
    args = parser.parse_args()

    model = init_langchain_model(args.llm, args.model_name)
    messages = [("system", "You are a helpful assistant. Please answer the question from the user."),
                ("human", args.query)]
    completion = model.invoke(messages)
    print(completion.content)

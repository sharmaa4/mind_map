# ai_services.py

import streamlit as st
import streamlit.components.v1 as components
import time
import json
from typing import List, Dict, Any

def build_context_prompt(
    user_query: str,
    document_context: str,
    note_context: str,
    conversation_history: List[Dict[str, Any]],
    max_context_messages: int
) -> str:
    """
    Builds a comprehensive prompt for the AI, including conversation history and retrieved context.

    Args:
        user_query (str): The current query from the user.
        document_context (str): Context retrieved from product documents.
        note_context (str): Context retrieved from user notes.
        conversation_history (List[Dict[str, Any]]): A list of previous user/assistant messages.
        max_context_messages (int): The number of recent messages to include in the prompt.

    Returns:
        str: The fully constructed prompt to be sent to the AI model.
    """
    # Start with conversation history if available
    if conversation_history:
        context_messages = ""
        # Get the most recent messages based on the slider value
        for msg in conversation_history[-max_context_messages:]:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            context_messages += f"{role_emoji} {msg['role'].capitalize()}: {msg['content']}\n"
        
        base_prompt = f"""Previous Conversation:
{context_messages}
Current Query: {user_query}"""
    else:
        base_prompt = f"User Query: {user_query}"

    # Add the context from retrieved documents
    base_prompt += f"\n\nDocument Context:\n{document_context}"
    
    # Add the context from personal notes if available
    if note_context:
        base_prompt += f"\n\nPersonal Notes Context:\n{note_context}"
        base_prompt += "\n\nPlease respond by synthesizing information from the document context and the user's personal notes."
    else:
        base_prompt += "\n\nPlease respond based on the provided document context and conversation history."
    
    return base_prompt

def create_streaming_puter_component(prompt: str, model: str = "gpt-4o-mini", stream: bool = True):
    """
    Creates a Streamlit HTML component to call the Puter.js AI and stream the response.

    Args:
        prompt (str): The complete prompt for the AI model.
        model (str): The AI model to use (e.g., "gpt-4o-mini").
        stream (bool): Whether to stream the response.
    """
    escaped_prompt = json.dumps(prompt)
    unique_id = f"puter-component-{int(time.time() * 1000)}"

    # This HTML and JavaScript code is embedded directly into the Streamlit app.
    # It calls the puter.ai.chat API and updates the UI in real-time.
    puter_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://js.puter.com/v2/"></script>
        <style>
            body {{ font-family: sans-serif; margin: 0; background: #f8f9fa; }}
            .container {{ background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e0e0e0; }}
            .result {{ white-space: pre-wrap; line-height: 1.6; color: #333; }}
            .loading {{ color: #666; }}
            .stats {{ margin-top: 1rem; font-size: 0.9em; color: #666; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div id="{unique_id}" class="result">
                <div class="loading">⏳ Processing with {model}...</div>
            </div>
            <div id="{unique_id}-stats" class="stats"></div>
        </div>
        <script>
            async function processQuery() {{
                const resultDiv = document.getElementById('{unique_id}');
                const statsDiv = document.getElementById('{unique_id}-stats');
                const prompt = {escaped_prompt};
                const modelName = "{model}";
                const streamingEnabled = {str(stream).lower()};
                
                try {{
                    const startTime = Date.now();
                    if (streamingEnabled) {{
                        resultDiv.innerHTML = '';
                        const response = await puter.ai.chat(prompt, {{ model: modelName, stream: true }});
                        let fullResponse = '';
                        for await (const chunk of response) {{
                            const content = chunk?.text || '';
                            if (content) {{
                                fullResponse += content;
                                resultDiv.innerText = fullResponse;
                            }}
                        }}
                        const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
                        statsDiv.innerText = `✅ Stream complete in: $\{totalTime}s`;
                    }} else {{
                        const response = await puter.ai.chat(prompt, {{ model: modelName, stream: false }});
                        resultDiv.innerText = response;
                        const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
                        statsDiv.innerText = `✅ Completed in: $\{totalTime}s`;
                    }}
                }} catch (error) {{
                    resultDiv.innerText = `❌ An error occurred: $\{error.message}`;
                    console.error("Puter.js error:", error);
                }}
            }}
            processQuery();
        </script>
    </body>
    </html>
    """
    components.html(puter_html, height=400)

def get_ai_response(
    user_query: str,
    document_context: str,
    note_context: str,
    conversation_history: List[Dict[str, Any]],
    model: str,
    enable_streaming: bool,
    enable_context: bool,
    max_context_messages: int
):
    """
    Orchestrates building the prompt and calling the Puter.js component to get the AI response.

    Args:
        user_query (str): The user's input query.
        document_context (str): Context from product documents.
        note_context (str): Context from personal notes.
        conversation_history (List[Dict[str, Any]]): Past conversation for context.
        model (str): The selected AI model.
        enable_streaming (bool): Flag to enable real-time streaming of the response.
        enable_context (bool): Flag to include conversation history.
        max_context_messages (int): The number of messages to include from history.
    """
    # Only use conversation history if the context feature is enabled
    history_to_use = conversation_history if enable_context else []
    
    full_prompt = build_context_prompt(
        user_query,
        document_context,
        note_context,
        history_to_use,
        max_context_messages
    )
    
    st.write("### 🤖 AI Response")
    if enable_context and history_to_use:
        st.info(f"🧠 Context Active: Remembering last {len(history_to_use)} messages.")
    if note_context:
        st.info("📝 Note Context: Including relevant personal notes in the response.")

    # Call the function to render the streaming component
    create_streaming_puter_component(full_prompt, model, enable_streaming)

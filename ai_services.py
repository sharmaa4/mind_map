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
    """
    if conversation_history:
        context_messages = ""
        for msg in conversation_history[-max_context_messages:]:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            context_messages += f"{role_emoji} {msg['role'].capitalize()}: {msg['content']}\n"
        
        base_prompt = f"""Previous Conversation:
{context_messages}
Current Query: {user_query}"""
    else:
        base_prompt = f"User Query: {user_query}"

    base_prompt += f"\n\nDocument Context:\n{document_context}"
    
    if note_context:
        base_prompt += f"\n\nPersonal Notes Context:\n{note_context}"
        base_prompt += "\n\nPlease respond by synthesizing information from the document context and the user's personal notes."
    else:
        base_prompt += "\n\nPlease respond based on the provided document context and conversation history."
    
    return base_prompt

def create_streaming_puter_component(prompt: str, model: str = "gpt-4o-mini", stream: bool = True):
    """
    Creates a Streamlit HTML component to call the Puter.js AI and stream the response.
    """
    escaped_prompt = json.dumps(prompt)
    unique_id = f"puter-component-{int(time.time() * 1000)}"

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
                        { # FIX: Escaped the curly braces for JavaScript variables # }
                        statsDiv.innerText = `✅ Stream complete in: ${{totalTime}}s`;
                    }} else {{
                        const response = await puter.ai.chat(prompt, {{ model: modelName, stream: false }});
                        resultDiv.innerText = response;
                        const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
                        { # FIX: Escaped the curly braces for JavaScript variables # }
                        statsDiv.innerText = `✅ Completed in: ${{totalTime}}s`;
                    }}
                }} catch (error) {{
                    { # FIX: Escaped the curly braces for JavaScript variables # }
                    resultDiv.innerText = `❌ An error occurred: ${{error.message}}`;
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
    """
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

    create_streaming_puter_component(full_prompt, model, enable_streaming)

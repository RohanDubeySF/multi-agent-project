from typing import Optional, Dict, Any, List
from langchain_core.messages import BaseMessage, get_buffer_string
from dotenv import load_dotenv
from prompt.router_agent_prompt import router_prompt
load_dotenv("utils/.env")
from loguru import logger
from utils.define_agent import model
from utils.llm_parser import clean_and_parse_agent_output_router




def router_agent(query:str,chat_history: Optional[List[BaseMessage]] = None,
    intermediate_data: Optional[Dict[str, Any]] = None)-> Dict[str, Any]:

    logger.info(f"[Router Agent] Received query: {query}")

    history_str= get_buffer_string(chat_history) if chat_history else " "

    logger.debug(f"[Router Agent] Chat History: {history_str}")
    logger.debug(f"[Router Agent] Intermediate Data: {intermediate_data}")


    prompt=router_prompt

    chain=prompt | model
    
    try:
        # Call the chain
        raw_response = chain.invoke({
            "query": query,
            "chat_history": history_str,
            "intermediate_data": str(intermediate_data)              
        })
        print(raw_response)
        logger.debug(f"[Router Agent] Raw LLM Output: {raw_response.content}")

        output=clean_and_parse_agent_output_router(raw_response.content)
        print(output)

        # Validate and fallback
        return {
            "next_node": output.get("next_node", "none"),
            "query": output.get("query", query),
            "reset_keys": output.get("reset_keys", []),
            "response": output.get("response", None)
        }

    except Exception as e:
        logger.error(f"[Router Agent] Failed to parse router output: {e}")
        return {
            "next_node": "none",
            "query": query,
            "reset_keys": [],
            "response": "Sorry, I could not understand that. Can you rephrase?"
        }

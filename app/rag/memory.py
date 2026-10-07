

from langchain_community.chat_message_histories import ChatMessageHistory

from app.monitoring.metrics import SESSIONS_CREATED

conversations={}

def get_or_create_memory(session_id:str)->ChatMessageHistory:
    
    if session_id  not in conversations:
        conversations[session_id]=ChatMessageHistory()
        SESSIONS_CREATED.inc()

    return conversations[session_id]
def get_chat_history_as_string(session_id:str)-> str:
    
   memory= get_or_create_memory(session_id)
   messages=memory.messages
   
   if not messages:
       return ""
   
   recent= messages[-6:]
   history_str="" 
   for msg in recent:
        # Handle both object and string formats
        if hasattr(msg, 'type'):
            role = "User" if msg.type == "human" else "Assistant"
            content = msg.content
        elif hasattr(msg, 'role'):
            role = "User" if msg.role == "human" else "Assistant"
            content = msg.content
        else:
            history_str += f"{msg}\n"
            continue
        history_str += f"{role}: {content}\n"
    
   return history_str.strip()

def save_to_memory(session_id: str,question: str, answer:str):
    
    memory= get_or_create_memory(session_id)
    memory.add_ai_message(answer)
    memory.add_messages(question)

def clear_memory(session_id: str):
    if session_id in conversations:
        del conversations[session_id]
        return True
    return False


   
       

   
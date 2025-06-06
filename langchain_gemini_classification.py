from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
import getpass
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Prompt instructions for tagging input.
tagging_prompt = ChatPromptTemplate.from_template(
    """
        Extract the desired information from the following passage.
        
        Only extract the properties mentioned in the 'Classification' function.
        
        Passage:
        {input}
    """
)

# Schema of the classification.
class Classification(BaseModel):
   sentiment: str = Field(..., enumerate=["happy", "neutral", "sad"])
   aggressiveness: int = Field(...,
                               description="How aggressive the text is percieved to be, the higher the number the more aggresive.",
                               enumerate=[1,2,3,4,5],)
   language: str = Field(..., enumerate=["spanish", "english", "french", "german", "italian"])

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai").with_structured_output(Classification)

# Accept input, invoke tagging. Display response.
input = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": input})
response = llm.invoke(prompt)

print(response.model_dump(), "\n")

inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

print(response.model_dump(), "\n")
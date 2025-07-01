from dataclasses import dataclass
import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,RunHooks,RunConfig,AgentHooks
from agents.tool import function_tool
from agents.run_context import RunContextWrapper
import asyncio


# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. Please ensure it is defined in your .env file."
    )

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@dataclass
class MyTestData:
  name: str
  age : int
  
test_data = MyTestData(name="Hamza",age=20)
  
# RunHooks
class myCustomRunhooks(RunHooks):
  async def on_start(self,ctx:RunContextWrapper[MyTestData],agent:Agent):
    print(f"Starting run for Agent {ctx.context.name} and age: {ctx.context.age}")
    
  async def on_end(self,ctx:RunContextWrapper[MyTestData],agent:Agent,output):
    print(f"Run completed for the agent {ctx.context.name} with age {ctx.context.age} and output: {output}")
    
    
class myCustomAgenthooks(AgentHooks):
  async def on_agent_start(self,ctx:RunContextWrapper[MyTestData],agent:Agent):
    print(f"Starting run for Agent {ctx.context.name} and age: {ctx.context.age}")
    
  async def on_agent_end(self,ctx:RunContextWrapper[MyTestData],agent:Agent,output):
    print(f"Run completed for the agent {ctx.context.name} with age {ctx.context.age} and output: {output}")

myagent = Agent(
  name="test agent",
  instructions="you are a helpful assistant" ,
  model=model,
)
result = Runner.run_sync(myagent,"what is Ai?",context=test_data,hooks=myCustomRunhooks(),run_config=config)        


        
     




import os
import chainlit as cl
from agents import Agent, Runner,WebSearchTool,input_guardrail,InputGuardrailTripwireTriggered,TResponseInputItem,GuardrailFunctionOutput,RunContextWrapper
from dotenv import load_dotenv
from pydantic import BaseModel
from openai.types.responses import ResponseTextDeltaEvent
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# summarizer_agent is an agent that creates concise summaries of academic resources provided by the Research Agent.
# It processes text from web results or files, extracts key points, and produces summaries (100 words or less) that help students understand the material quickly.
summarizer_agent: Agent = Agent(
    name="Summarizer Assistant",
    instructions="""
    You are a Summarizer Agent, an AI designed to create concise, clear summaries of academic resources provided by the Research Agent. Your goal is to process text from web results or files, extract key points, and produce summaries (100 words or less) that help students understand the material quickly. Save summaries to 'study_summaries.txt' in Markdown format. Ensure accuracy, skip irrelevant content, and use simple language. If a resource cannot be summarized, note it in the output.
    """,
    handoff_description="Summarizer Assistant",
)

# Research Agent to gather information from the web based on topics and deadlines provided by the scheduler agent.
# It will filter out irrelevant content and transfer the relevant data to the summarizer agent.
researcher_agent: Agent = Agent(
    name="Research Assistant",
    instructions="""
    I am a researcher agent. Your job is to oversee the scheduler agent, which will send topics and deadlines. Using the web search tool, you will use the responses API to search the internet. You will collect data through videos and other resources, filtering out content that is relevant to the user's topics and deadlines. Then, you will transfer the filtered data to the summarizer agent
    """,
    tools=[WebSearchTool()],
    handoffs=[summarizer_agent],
    handoff_description="Research Assistant",
)

# Pydantic Class for Guardrail Output
class isStudyOuput(BaseModel):
  is_study_input: bool
  reason: str
  study_topic: str
  deadline: str

# Guardrail Agent to validate user input for study-related tasks
guardrail_agent: Agent = Agent(
    name="Guardrail Check",
    instructions="""
    You are a Guardrails Agent designed to validate user study-related requests. Your goal is to ensure that the user's input is clear, ethical, and aligned with the purpose of a Personal Study Assistant. For every user input, follow these steps: Pay attention to keywords and phrases that indicate study-related tasks. Common keywords include "study," "learn," "prepare," "homework," "assignment," "deadline," "exam," and specific subjects such as "math," "science," "programming." Check whether the user has provided specific study topics and a clear deadline. The topics should be academic or educational, and the deadline should be a future date.
    """,
    output_type=isStudyOuput
)


# This function is used to validate the user's input and ensure it meets the criteria for study-related tasks.
@input_guardrail
async def study_guardrail(ctx:RunContextWrapper[None],agent:Agent,input: str | list[TResponseInputItem])->GuardrailFunctionOutput:
  result = await Runner.run(guardrail_agent,input,context=ctx.context)
  
  return GuardrailFunctionOutput(
    output_info = result.final_output,
    tripwire_triggered = result.final_output.is_study_input is False,
  )
  
# Scheduler Agent to create a study plan based on user input
# It will hand off the topics and deadlines to the research assistant for further processing.
scheduler_agent: Agent = Agent(
    name="Scheduler Assistant",
    instructions="""
    You are a study scheduler assistant designed to create a plan based on the user's study topics and deadlines. The plan should outline how the user will complete each topic by the given deadline. After completing this task, you need to hand off the topic and deadlines to the research assistant.
    """,
    handoffs=[researcher_agent],
    handoff_description="Scheduler Assistant",
    input_guardrails=[study_guardrail]
)
    
@cl.on_chat_start
async def on_chat_start():
  cl.user_session.set("history", [])
  await cl.Message(content="## Welcome to the Personal Study Scheduler!").send()
  await cl.Message(content="Please provide your study topics and deadlines.").send()


@cl.on_message
async def handle_chat(message: cl.Message):
  history = cl.user_session.get("history")
  msg = cl.Message(content="Reasoning....")
  await msg.send()
  
  history.append({"role": "user", "content": message.content})
  try:
      result = Runner.run_streamed(scheduler_agent,history)
      async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance (event.data,ResponseTextDeltaEvent):
          await msg.stream_token(event.data.delta)
        
      
  except InputGuardrailTripwireTriggered as e:
    print(f"Tripwire triggered: {e}")
    
  history.append({"role": "assistant", "content": result.to_input_list})
  cl.user_session.set("history", history)
  await cl.Message(content=result.final_output).send()
  



# =======================================================
# feedback_engine.py | RAG + Smart Fallback (Final Version)
# =======================================================
import numpy as np
from langchain import PromptTemplate, LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain_community.tools import DuckDuckGoSearchRun
from transformers import pipeline

# ---------------------------------------------------------
# 🧠 FREE RAG SYSTEM — Web-Enhanced + AI-Generated Feedback
# ---------------------------------------------------------

# ✅ Step 1: Setup Web Retriever (DuckDuckGo)
try:
    search = DuckDuckGoSearchRun()
except Exception as e:
    print("⚠ DuckDuckGo setup failed:", e)
    search = None

# ✅ Step 2: Load Hugging Face Model (Flan-T5)
try:
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  # upgraded for stronger reasoning
        max_length=256
    )
    llm = HuggingFacePipeline(pipeline=generator)
except Exception as e:
    print("⚠ Model setup failed:", e)
    llm = None


# ✅ Step 3: Web Retrieval
def cloud_retrieve(query: str) -> str:
    """Fetch relevant info from the web for the given driving condition."""
    if not search:
        return "Retriever not available (no internet connection)."
    try:
        results = search.run(query)
        return results[:500] if results else "No web results found."
    except Exception as e:
        return f"[Retriever Error] {e}"


# ✅ Step 4: Generate feedback using RAG
def generate_feedback_with_llm(condition, base_message, spd, acc, brk):
    """Generate a personalized feedback message using RAG pipeline."""
    query = f"{condition} driving safety tips"
    web_data = cloud_retrieve(query)

    if not llm:
        return f"{base_message} (⚠ Offline mode: could not generate AI feedback.)"

    # Prompt Template
    prompt = PromptTemplate(
        input_variables=["condition", "base_message", "spd", "acc", "brk", "web_data"],
        template="""
        You are Alwa — a friendly and intelligent AI driving assistant.

        Situation: {condition}
        System note: {base_message}

        Driving stats:
        - Speed: {spd} km/h
        - Acceleration: {acc}
        - Brake Pressure: {brk}

        Reference for understanding (do NOT include in your answer):
        {web_data}

        Write a single, friendly message for the driver.
        It should:
        - Be under 3 sentences.
        - Provide helpful appreciation or safety advice.
        - Sound natural and encouraging.
        - NOT include words like “reference”, “using this information”, or “situation”.
        Respond only with the final message.
        """
    )

    chain = LLMChain(prompt=prompt, llm=llm)
    response = chain.run(
        condition=condition,
        base_message=base_message,
        spd=spd,
        acc=acc,
        brk=brk,
        web_data=web_data
    ).strip()

    # 🔧 Clean up any stray text
    bad_phrases = [
        "Using this information",
        "Respond only",
        "Provide helpful",
        "Situation:",
        "Reference for understanding",
        "Be under 3 sentences",
        "System note"
    ]
    for phrase in bad_phrases:
        if phrase in response:
            response = response.split(phrase)[0].strip()

    # ✅ Smart fallback — if the model fails to generate a valid response
    if len(response.split()) < 4 or "appreciation" in response.lower():
        if condition == "smooth_driving":
            response = "Excellent control! Smooth driving like this saves fuel and keeps you safe."
        elif condition == "harsh_braking":
            response = "Brake gradually and maintain distance — sudden braking wears out tires and can be risky."
        elif condition == "rapid_acceleration":
            response = "Avoid sudden acceleration. Smooth speed control improves safety and comfort."
        elif condition == "over_speeding":
            response = "You’re driving fast. Reduce speed to stay safe and avoid accidents."
        else:
            response = base_message

    return response


# ✅ Step 5: Analyze Driving Data + Call RAG
def analyze_driving(ax, ay, az, spd, brk, acc):
    BRAKE_THRESHOLD = -3.0
    ACCEL_THRESHOLD = 3.0
    SPEED_LIMIT = 80.0

    if ax < BRAKE_THRESHOLD or brk > 10:
        event = "harsh_braking"
        base_message = "⚠ Harsh braking detected. Please brake smoothly."
    elif ax > ACCEL_THRESHOLD or acc > 10:
        event = "rapid_acceleration"
        base_message = "🚗 Rapid acceleration detected. Ease off the accelerator."
    elif spd > SPEED_LIMIT:
        event = "over_speeding"
        base_message = "⚠ Over-speeding detected. Please reduce your speed."
    else:
        event = "smooth_driving"
        base_message = "✅ Smooth driving. Keep it up!"

    personalized_feedback = generate_feedback_with_llm(
        condition=event,
        base_message=base_message,
        spd=spd,
        acc=acc,
        brk=brk,
    )

    return {"event": event, "message": personalized_feedback}
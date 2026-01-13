from config.settings import llm

def song_generation_agent(state):
    try:
        if not state["enriched_eponyms"]:
            state["medical_song"] = "No valid eponyms found."
            return state

        eponyms = "\n".join(
            f"- {e['name']}: {e['description']}"
            for e in state["enriched_eponyms"]
        )

        prompt = f"""
Write a creative, medically accurate song using these eponyms.
Include verses and chorus.

{eponyms}
"""
        response = llm.invoke(prompt)
        state["medical_song"] = response.content
        return state

    except Exception as e:
        state["error"] = f"Song generation failed: {e}"
        return state

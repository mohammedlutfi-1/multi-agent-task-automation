
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient


async def main():
    model_client = OllamaChatCompletionClient(
        model="qwen2.5:1.5b",
        host="http://localhost:11434",
    )

    planner = AssistantAgent(
        name="planner",
        model_client=model_client,
        system_message=(
            "You are a planning agent. "
            "Break user tasks into three clear steps."
        ),
    )

    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        system_message=(
            "You are a writing agent. "
            "Complete the task using the plan provided. "
            "Write clear, useful content."
        ),
    )
    
    reviewer = AssistantAgent(
        name="reviewer",
        model_client=model_client,
        system_message=(
            "You are a strict content reviewer. "
            "Evaluate the draft against the original task. "
            "Identify factual errors, missing information, "
            "unclear explanations, repetition, and unsupported claims. "
            "Do not give generic praise. "
            "Provide specific, actionable corrections. "
            "If the draft is already good, identify remaining limitations "
            "or state that no major corrections are needed."
        ),
    )

    try:
        task = "Explain how AI is used in cybersecurity."

        # Agent 1: Plan
        plan_result = await planner.run(task=task)
        plan = str(plan_result.messages[-1].content)

        print("\n=== PLANNER AGENT ===\n")
        print(plan)

        # Agent 2: Write
        write_result = await writer.run(
            task=f"Original task: {task}\n\nPlan:\n{plan}"
        )
        draft = str(write_result.messages[-1].content)

        print("\n=== WRITER AGENT ===\n")
        print(draft)

        # Agent 3: Review
        review_result = await reviewer.run(
            task=f"Review this draft:\n\n{draft}"
        )
        review = str(review_result.messages[-1].content)

        print("\n=== REVIEWER AGENT ===\n")
        print(review)

        
        # Step 4: Writer Agent revises using reviewer feedback
        revision_result = await writer.run(
            task=(
                f"Original task:\n{task}\n\n"
                f"Previous draft:\n{draft}\n\n"
                f"Reviewer feedback:\n{review}\n\n"
                "Improve the previous draft based on the feedback. "
                "Return only the revised final answer."
            )
        )

        final_output = str(
            revision_result.messages[-1].content
        )

        print("\n=== FINAL IMPROVED OUTPUT ===\n")
        print(final_output)


    finally:
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

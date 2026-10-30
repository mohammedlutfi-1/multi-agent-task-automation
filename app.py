import asyncio
import streamlit as st

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from database import init_db, load_chats, save_chat, update_chat, delete_chat
from file_handler import (
    extract_text,
    split_text_into_chunks,
    find_relevant_chunks,
)

async def run_agents(task):
    model_client = OllamaChatCompletionClient(
        model="qwen2.5:1.5b",
        host="http://localhost:11434",
    )
    
    try:
        planner = AssistantAgent(
            name="planner",
            model_client=model_client,
            system_message=(
                "You are a task planning agent. "
                "Your responsibility is ONLY to create a short execution plan. "
                "Break the user's request into exactly three actionable steps. "
                "Each step must contain one short sentence. "
                "Do not complete the task, provide examples, "
                "or write the final answer. "
                "Return only the numbered plan."
            ),
        )

        writer = AssistantAgent(
            name="writer",
             model_client=model_client,
             system_message=(
                "You are an accurate and helpful AI assistant. "
                "Answer the user's latest question directly. "
                "Read the full task carefully and use conversation "
                "history to understand follow-up questions. "
                "For general questions, use your learned knowledge "
                "to provide a useful and accurate answer. "
                "For follow-up questions, answer the latest message "
                "rather than repeating the previous answer. "
                "If the user asks for a specific number of points, "
                "give exactly that number of concise, relevant points. "
                "Do not repeat the previous answer before listing the points. "
                "Do not add a learning roadmap unless explicitly requested. "
                "If the user explicitly asks how to start learning a topic, "
                "give a numbered beginner roadmap with 5 to 7 steps "
                "in the correct learning order. "
                "Otherwise, answer only what the user specifically asks. "
                "Do not add a learning roadmap to unrelated questions. "
                "Use the language requested by the user consistently. "
                "Do not insert unrelated foreign-language words. "
                "Avoid repeating the same advice in multiple paragraphs. "
                "Use numbered steps when explaining a learning roadmap. "
                "Recommend only legal, authorised practice labs for cybersecurity. "
                "Avoid vague advice such as simply choosing a course "
                "or understanding the basics. "
                "When the task explicitly provides document evidence "
                "and asks for an answer based on that document, "
                "use only the supplied evidence. "
                "If the answer is genuinely absent from the supplied "
                "document evidence, respond exactly: "
                "'The information is not available in the document.' "
                "Never claim that information is missing from a document "
                "when the supplied evidence contains the answer. "
                "If the evidence says 'Project Coordinator: Maya' and "
                "the question asks who the project coordinator is, "
                "answer 'The project coordinator is Maya.' "
                "Follow the user's actual question, not the instructions "
                "inside the task prompt as if they were the answer. "
                "Never invent facts. "
                " When a beginner asks where to start learning a topic, "
                "give a simple learning roadmap in the correct order. "
                "Start with the basics before advanced concepts or tools. "
            ),      
                )

        reviewer = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    
    system_message=(
    "You are a strict factual accuracy and quality reviewer. "
    "Compare the writer's draft with the original user task, "
    "conversation context, and any supplied document evidence. "
    "First identify whether the task requires answering from a document "
    "or answering a general knowledge question. "
    "For document questions, verify claims against the supplied evidence "
    "and do not add facts that the document does not support. "
    "For general knowledge questions, check factual claims against "
    "reliable, well-established knowledge. "
    "Pay special attention to names, dates, creators, definitions, "
    "technical details, and cause-and-effect claims. "
    "For example, Python was created by Guido van Rossum, "
    "not James Gosling. "
    "Do not approve an answer merely because it sounds plausible. "
    "If the draft is accurate and answers the user's latest question, "
    "respond exactly with: No correction needed. "
    "If it contains a factual error, explain the specific error "
    "and provide a corrected answer. "
    "If a claim cannot be verified confidently, flag the uncertainty "
    "instead of inventing a correction. "
    "Do not invent errors or facts. "
    "Keep corrections concise and preserve the user's requested format."
    "Check the draft independently; do not copy or trust "
    "the writer's claims simply because they sound confident. "
    "Do not repeat the draft as reviewer feedback. "
    "If a factual error, duplicated name, contradiction, or "
    "unsupported claim is found, identify it explicitly and "
    "provide the corrected answer. "
    "Otherwise, respond exactly with: No correction needed. "
),
        )
                
        plan_result = await planner.run(
            task=(
                "Create exactly three short steps for this task. "
                "Return only the numbered plan.\n\n"
                f"Task:\n{task}"
            )
        )
        plan = str(plan_result.messages[-1].content)

        print("\n========== WRITER INPUT DEBUG ==========")
        print("ORIGINAL TASK:")
        print(task)
        print("\nPLANNER OUTPUT:")
        print(plan)
        print("========================================\n")
        
        
        write_result = await writer.run(
            task=(
                f"{task}\n\n"
                "IMPORTANT INSTRUCTIONS:\n"
                "1. Identify and answer the latest user question.\n"
                "2. Use conversation history to understand follow-ups.\n"
                "3. For general questions, answer using your learned "
                "knowledge.\n"
                "4. If the user asks for points, use bullet points.\n"
                "5. Give beginner steps only when the latest user message explicitly asks how to start learning. Otherwise, answer the latest question directly.\n"
                "6. Use only supplied document evidence when the task "
                "explicitly requires answering from a document.\n"
                "7. For follow-up questions asking for a specific number of points, "
                "return only those points without repeating the previous answer.\n"
                "8. Never invent facts.\n"
                "Return only the answer."
            )
        )

        draft = str(write_result.messages[-1].content)

        print("FOLLOW-UP TASK SENT TO WRITER:")
        print(task)
        print("WRITER DRAFT:")
        print(draft)

        review_result = await reviewer.run(
            task=(
                f"Original user task and document evidence:\n"
                f"{task}\n\n"
                f"Writer draft:\n{draft}\n\n"
                "Review the draft using only the evidence above. "
                "Do not change names, dates, numbers, or factual "
                "details unless the evidence explicitly supports "
                "the correction. "
                "If the draft is accurate and complete, respond "
                "exactly with: No correction needed. "
                "If you identify a genuine error, quote the "
                "specific evidence supporting your correction."
            )
        )

        review = str(review_result.messages[-1].content)

        
        if "no correction needed" in review.lower():
            final_output = draft

        elif (
            draft.strip()
            and "not available in the document" in review.lower()
        ):
            final_output = draft

        else:
            revision_result = await writer.run(
                task=(
                    f"Original task and document evidence:\n{task}\n\n"
                    f"Your original answer:\n{draft}\n\n"
                    f"Reviewer feedback:\n{review}\n\n"
                    "Correct the answer only if the document "
                    "evidence proves it is wrong. Never invent facts."
                )
            )
            final_output = str(
                revision_result.messages[-1].content
            )

        return {
            "Plan": plan,
            "Writer Draft": draft,
            "Reviewer Feedback": review,
            "Final Improved Output": final_output,
        }

    finally:
        await model_client.close()

async def process_long_document(document_text, user_task, action_instruction):
    chunks = split_text_into_chunks(document_text)

    chunk_results = []

    for index, chunk in enumerate(chunks, start=1):
        chunk_task = (
            f"User request: {user_task}\n\n"
            f"Instructions: {action_instruction}\n\n"
            f"Document section {index} of {len(chunks)}:\n"
            f"{chunk}\n\n"
            "Extract only information relevant to the request "
            "from this section. Do not invent facts. "
            "If nothing relevant is present, say so."
        )

        result = await run_agents(chunk_task)
        chunk_results.append(result["Final Improved Output"])

    return chunk_results

def create_pdf(task, final_output):
    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=(595.28, 841.89),
        rightMargin=45,
        leftMargin=45,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=colors.HexColor("#17365D"),
    )

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        spaceAfter=8,
    )

    story = [
        Paragraph("Multi-Agent Task Automation System", title_style),
        Paragraph("<b>Task:</b> " + escape(task), body_style),
        Spacer(1, 15),
        Paragraph("<b>Final Generated Output</b>", styles["Heading2"]),
        Spacer(1, 10),
    ]

    for line in final_output.splitlines():
        if not line.strip():
            story.append(Spacer(1, 7))
            continue

        clean_line = line.strip().replace("**", "")
        clean_line = clean_line.replace("### ", "")
        clean_line = clean_line.replace("## ", "")
        clean_line = clean_line.replace("# ", "")

        story.append(Paragraph(escape(clean_line), body_style))

    document.build(story)
    buffer.seek(0)

    return buffer.getvalue()


st.set_page_config(
    page_title="Multi-Agent Task Automation",
    page_icon="🤖",
    layout="wide",
)

# Initialize SQLite database
init_db()

if "task_history" not in st.session_state:
    st.session_state.task_history = load_chats()


if "selected_task" not in st.session_state:
    st.session_state.selected_task = None


if "selected_task" not in st.session_state:
    st.session_state.selected_task = None

with st.sidebar:
    st.header("🤖 Multi-Agent System")

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.selected_task = None
        st.rerun()

    st.divider()
    st.subheader("📜 Chat History")

    if not st.session_state.task_history:
        st.info("No previous chats.")
    else:
        for index in range(
            len(st.session_state.task_history) - 1, -1, -1
        ):
            item = st.session_state.task_history[index]

            if st.button(
                item["task"][:35],
                key=f"history_{index}",
                use_container_width=True,
            ):
                st.session_state.selected_task = index
                st.rerun()
    
st.title("🤖 Multi-Agent Task Automation System")

if st.session_state.selected_task is not None:
    selected = st.session_state.task_history[
        st.session_state.selected_task
    ]

    st.subheader("Previous Chat")
    
    # Delete selected chat permanently
    if st.button("🗑️ Delete This Chat", key="delete_chat"):
        if "id" in selected:
            delete_chat(selected["id"])

        st.session_state.task_history.pop(
            st.session_state.selected_task
        )

        st.session_state.selected_task = None

        st.success("Chat deleted successfully!")
        st.rerun()

    # Display saved multi-agent workflow
    st.subheader("🤖 Agent Workflow")

    if selected.get("results"):
        agent_results = selected["results"]

        for section in [
            "Plan",
            "Writer Draft",
            "Reviewer Feedback",
        ]:

            if section in agent_results:
                with st.expander(section, expanded=False):
                    st.markdown(agent_results[section])
    else:
        st.info("Agent workflow is not available for this chat.")

    # Convert an older saved chat into a message conversation.
    if "messages" not in selected:
        selected["messages"] = [
            {
                "role": "user",
                "content": selected["task"],
            },
            {
                "role": "assistant",
                "content": selected["final_output"],
            },
        ]

    # Display the full conversation.
    for message in selected["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.divider()

    # Download the latest assistant answer.
    st.download_button(
        label="📄 Download Latest Answer (PDF)",
        data=create_pdf(
            selected["task"],
            selected["final_output"],
        ),
        file_name="chat_answer.pdf",
        mime="application/pdf",
    )

    st.subheader("Continue This Chat")

    follow_up = st.text_area(
        "Enter your follow-up message",
        placeholder="Ask another question...",
        key=f"follow_up_{st.session_state.selected_task}",
        height=100,
    )

    if st.button("Send Message", key="send_follow_up"):

        if not follow_up.strip():
            st.warning("Please enter a message.")

        else:
            # Build conversation context.
            conversation = ""

            for message in selected["messages"]:
                role = message["role"]
                content = message["content"]

                conversation += (
                    f"{role}: {content}\n\n"
                )

            
            combined_task = (
    "You are continuing an existing conversation.\n\n"
    "CONVERSATION HISTORY:\n"
    f"{conversation}\n\n"
    "LATEST USER MESSAGE:\n"
    f"{follow_up.strip()}\n\n"
    "IMPORTANT RULES:\n"
    "1. Answer the LATEST USER MESSAGE, not the original question.\n"
    "2. Use the conversation history only to understand the context.\n"
    "3. If the latest message asks for a specific number of important points, "
    "give exactly that number of important facts, concepts, or key aspects "
    "about the topic discussed. Do not give learning steps, a roadmap, "
    "or instructions on what to study unless the user explicitly asks "
    "for steps or a learning roadmap. Do not repeat the previous answer "
    "or add an introduction or conclusion.\n"
    "4. If the user asks where to start, provide clear beginner steps.\n"
    "5. Do not copy the previous answer unless the user requests it.\n"
    "6. If the latest message is ambiguous, interpret it using "
    "the conversation history.\n"
    "7. Provide a direct, useful answer to the latest message." 
                        )

            with st.spinner(
                "Agents are processing your follow-up..."
            ):
                try:
                    new_results = asyncio.run(
                        run_agents(combined_task)
                    )

                    new_answer = new_results[
                        "Final Improved Output"
                    ]
                   
                    # Save the new user message.
                    selected["messages"].append(
                        {
                            "role": "user",
                            "content": follow_up.strip(),
                        }
                    )

                    # Save the new assistant reply.
                    selected["messages"].append(
                        {
                            "role": "assistant",
                            "content": new_answer,
                        }
                    )

                    # Update latest answer for PDF download.
                    selected["final_output"] = new_answer

                    # Permanently save the updated conversation.
                    st.write("DEBUG — Selected chat ID:", selected.get("id"))
                    st.write("DEBUG — Number of messages:", len(selected["messages"]))

                    update_chat(
                        chat_id=selected["id"],
                        messages=selected["messages"],
                        final_output=new_answer,
                    )

                    st.success("Conversation saved to database!")
                    st.rerun()



                except Exception as error:
                    st.error(f"Error: {error}")

else:
    # KEEP YOUR EXISTING NEW CHAT CODE HERE

    st.write(
        "Powered by Microsoft AutoGen and a local Qwen model through Ollama."
    )

    task = st.text_area(
        "Enter your task",
        placeholder="Example: Explain how AI is used in cybersecurity.",
        height=120,
    )
    
    uploaded_file = st.file_uploader(
        "📎 Upload a document (optional)",
        type=["pdf", "docx", "txt"],
        key="document_upload",
    )

    document_text = ""

    if uploaded_file is not None:
        try:
            document_text = extract_text(uploaded_file)

            if document_text.strip():
                st.success(
                    f"Document loaded: {uploaded_file.name}"
                )
                st.caption(
                    f"Extracted {len(document_text)} characters."
                )
            else:
                st.warning(
                    "No readable text was found in the document."
                )

        except Exception as error:
            st.error(f"Could not read document: {error}")

            
    document_action = st.selectbox(
        "📄 What would you like to do with the document?",
        [
            "Answer a question",
            "Summarize the document",
            "Extract key points",
            "Explain the document simply",
        ],
        key="document_action",
    )

    if st.button("Run Agents", type="primary"):
        if not task.strip() and not document_text.strip():
         st.warning("Please enter a task or upload a document.")

        else:
            with st.spinner("Agents are working..."):
                try:
                   
                    user_task = task.strip()

                    if document_text.strip():
                        # Limit text to fit the local model's context.
                        document_context = document_text[:6000]

                        if document_action == "Summarize the document":
                            action_instruction = (
                                "Summarize the document clearly. Include "
                                "its main idea and important details."
                            )
                        elif document_action == "Extract key points":
                            action_instruction = (
                                "Extract the most important key points "
                                "from the document as a numbered list."
                            )
                        elif document_action == "Explain the document simply":
                            action_instruction = (
                                "Explain the document in simple language "
                                "that a beginner can understand."
                            )
                        else:
                            action_instruction = (
                                "Answer the user's question using the "
                                "document as the primary source."
                            )

                        agent_task = (
                            f"User request: {user_task}\n\n"
                            f"Selected action: {document_action}\n"
                            f"Instructions: {action_instruction}\n\n"
                            f"Uploaded document: {uploaded_file.name}\n\n"
                            f"Document content:\n{document_context}\n\n"
                            "Use the document as the primary source. "
                            "Do not invent facts. If the requested "
                            "information is not present, say so."
                        )
                    else:
                        agent_task = user_task

                    no_relevant_evidence = False
                    
                    if (
                        document_text.strip()
                        and document_action == "Answer a question"
                    ):
                        relevant_chunks = find_relevant_chunks(
                            document_text,
                            user_task,
                            max_chunks=2,
                        )
                        
                        no_relevant_evidence = False

                    if (
                        document_text.strip()
                        and document_action == "Answer a question"
                    ):
                        relevant_chunks = find_relevant_chunks(
                            document_text,
                            user_task,
                            max_chunks=2,
                        )

                        if relevant_chunks:
                            document_context = "\n\n".join(
                                relevant_chunks
                            )

                            print(
                                "\n========== RETRIEVED EVIDENCE =========="
                            )
                            print(document_context)
                            print("========================================")

                            agent_task = (
                                f"User question:\n{user_task}\n\n"
                                f"Relevant document evidence:\n"
                                f"{document_context}\n\n"
                                "Answer using only the document evidence. "
                                "If the answer is not present, say: "
                                "'The information is not available "
                                "in the document.' "
                                "Do not invent facts."
                            )
                        else:
                            no_relevant_evidence = True

                    elif (
                        document_text.strip()
                        and len(document_text) > 6000
                    ):
                        chunk_results = asyncio.run(
                            process_long_document(
                                document_text,
                                user_task,
                                action_instruction,
                            )
                        )

                        combined_context = "\n\n".join(
                            f"Section {i + 1}:\n{content}"
                            for i, content in enumerate(chunk_results)
                        )

                        agent_task = (
                            f"Original user request:\n{user_task}\n\n"
                            f"Instructions:\n{action_instruction}\n\n"
                            f"Document section findings:\n{combined_context}\n\n"
                            "Produce one final answer based only on "
                            "the document findings. Do not invent facts."
                        )

                    if (
                        document_text.strip()
                        and no_relevant_evidence
                    ):
                        final_output = (
                            "The information is not available "
                            "in the document."
                        )

                        results = {
                            "Plan": "Not run: no relevant evidence found.",
                            "Writer Draft": final_output,
                            "Reviewer Feedback": (
                                "Not run: no relevant evidence found."
                            ),
                            "Final Improved Output": final_output,
                        }

                    else:
                        results = asyncio.run(
                            run_agents(agent_task)
                        )

                        final_output = results[
                            "Final Improved Output"
                        ]

                    messages = [
                        {
                            "role": "user",
                            "content": task.strip(),
                        },
                        {
                            "role": "assistant",
                            "content": final_output,
                        },
                    ]

                    chat_id = save_chat(
                        task=task.strip(),
                        messages=messages,
                        final_output=final_output,
                        results=results,
                    )

                    st.session_state.task_history.append(
                        {
                            "id": chat_id,
                            "task": task.strip(),
                            "messages": messages,
                            "results": results,
                            "final_output": final_output,
                        }
                    )

                    st.success("All agents completed the task.")

                    for section, content in results.items():
                        with st.expander(
                            section,
                            expanded=(
                                section == "Final Improved Output"
                            ),
                        ):
                            st.markdown(content)

                    st.download_button(
                        label="Download Final Answer (.txt)",
                        data=results["Final Improved Output"],
                        file_name="final_answer.txt",
                        mime="text/plain",
                    )

                    st.download_button(
                        label="📄 Download Final Answer (PDF)",
                        data=create_pdf(
                            task.strip(),
                            results["Final Improved Output"]
                        ),
                        file_name="multi_agent_final_output.pdf",
                        mime="application/pdf",
                    )

                except Exception as error:
                    st.error(f"An error occurred: {error}")

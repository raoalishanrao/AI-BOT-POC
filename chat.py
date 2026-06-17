"""CLI counseling chatbot with session memory."""

from src.chat.counselor import chat, start_session
from src.utils.logger import setup_logger

log = setup_logger("chat")


def main():
    log.info("Iqra University AI Counselor (type 'exit' to quit, 'new' for fresh session)")
    session = start_session()
    print(f"\nAssistant: {session.reply}")
    print(f"[session: {session.session_id[:8]}…]")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "new":
            session = start_session()
            print(f"\nAssistant: {session.reply}")
            print(f"[session: {session.session_id[:8]}…]")
            continue

        response = chat(session.session_id, question)
        print(f"\nAssistant: {response.reply}")
        if response.recommended_programs:
            print(f"\n[recommended: {', '.join(response.recommended_programs)}]")
        if response.lead_status in {"warm", "interested", "captured"}:
            print(f"[lead status: {response.lead_status}]")
            for cta in response.ctas[:3]:
                print(f"  → {cta['label']}: {cta['url']}")


if __name__ == "__main__":
    main()

"""
sim_tool.run_directive
──────────────────────
CLI entrypoint to run a PIDirective from a JSON file.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from comms.pi_email import PIMailbox
from sim_tool.research_directive import PIDirective, DirectiveRunner, AgentLLMConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PIDirective from JSON")
    parser.add_argument("directive_file", help="Path to JSON directive")
    args = parser.parse_args()

    with open(args.directive_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    llm_config = data.get("llm")
    if llm_config:
        llm = AgentLLMConfig(**llm_config)
    else:
        llm = None

    directive = PIDirective(
        directive_id=data.get("directive_id", "dir-001"),
        pi_name=data.get("pi_name", "PI"),
        instruction=data.get("instruction", ""),
        topic_hint=data.get("topic_hint", ""),
        phases=data.get("phases", []),
        output_dir=data.get("output_dir", "./output"),
        llm=llm
    )

    mailbox = PIMailbox(output_dir=Path(directive.output_dir))
    runner = DirectiveRunner(directive, mailbox)
    runner.run_all_phases()

    print("\n📬 Mailbox contents:")
    for email in mailbox.all_emails():
        print(f"FROM: {email.from_agent} <{email.from_addr}>")
        print(f"TO:   {email.to_name}")
        print(f"SUBJ: {email.subject}")
        print("-" * 40)


if __name__ == "__main__":
    main()

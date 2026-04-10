DEDUP_PROMPT = """Are these two messages about the same news event?

Message A:
---
{message_a}
---

Message B:
---
{message_b}
---

Return ONLY valid JSON:
{{"is_duplicate": true/false, "reason": "<brief explanation>"}}
"""

CLASSIFY_PROMPT = """Classify this message and extract key entities.

CATEGORIES:
{categories_block}

MESSAGE:
---
{message_content}
---

Return ONLY valid JSON:
{{
  "categories": {{"<category_name>": <float 0.0-1.0>, ...}},
  "relevance": <float 0.0-1.0>,
  "summary": "<1-2 sentence factual summary>",
  "entities": [
    {{"name": "<entity name>",
      "type": "<model|company|person|technology|event|regulation|product|concept>"}}
  ]
}}

Category scoring: 0.0=unrelated, 0.5=tangential, 0.8=strong match, 1.0=core topic.
A message CAN score high in multiple categories.
Relevance: 1.0=breaking news, 0.7=notable, 0.4=routine, 0.1=noise/spam.
Summary: key fact or claim, neutral tone, include names and numbers.
Entities: extract ALL named entities (people, companies, models, products, regulations).
"""

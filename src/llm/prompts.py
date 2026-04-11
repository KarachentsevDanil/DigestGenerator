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

EXTRACT_KNOWLEDGE_PROMPT = """Extract entity relationships from this news message.

ENTITY TYPES (use ONLY these):
PERSON, ORGANIZATION, MODEL, TECHNOLOGY, PRODUCT, EVENT, REGULATION, LOCATION, CONCEPT, DATASET

RELATION TYPES (use ONLY these):
Structural: CREATED_BY, WORKS_AT, SUBSIDIARY_OF, HEADQUARTERED_IN, MEMBER_OF, FUNDED_BY
Competitive: COMPETES_WITH, OUTPERFORMS, BASED_ON, SUCCESSOR_OF, ALTERNATIVE_TO
Event-driven: ANNOUNCED, ACQUIRED, LAUNCHED, PARTNERED_WITH, INVESTED_IN
Stance: SUPPORTS, OPPOSES, REGULATES
Catch-all: RELATED_TO

EXAMPLES:

Text: "Google DeepMind announced Gemma 3, a new open-weight language model that outperforms Meta's Llama 3.1 on key benchmarks. The model was trained on TPU v5 hardware at Google's data centers."
Output:
{{
  "entities": [
    {{"name": "Google DeepMind", "type": "ORGANIZATION"}},
    {{"name": "Gemma 3", "type": "MODEL"}},
    {{"name": "Meta", "type": "ORGANIZATION"}},
    {{"name": "Llama 3.1", "type": "MODEL"}},
    {{"name": "TPU v5", "type": "TECHNOLOGY"}}
  ],
  "relations": [
    {{"subject": "Google DeepMind", "predicate": "ANNOUNCED", "object": "Gemma 3", "confidence": 0.95}},
    {{"subject": "Gemma 3", "predicate": "CREATED_BY", "object": "Google DeepMind", "confidence": 0.95}},
    {{"subject": "Gemma 3", "predicate": "OUTPERFORMS", "object": "Llama 3.1", "confidence": 0.85}},
    {{"subject": "Llama 3.1", "predicate": "CREATED_BY", "object": "Meta", "confidence": 0.90}},
    {{"subject": "Gemma 3", "predicate": "BASED_ON", "object": "TPU v5", "confidence": 0.80}}
  ]
}}

Text: "The EU Parliament passed the AI Act, which will regulate foundation models like GPT-5 and Claude. OpenAI CEO Sam Altman said the company supports the regulation."
Output:
{{
  "entities": [
    {{"name": "EU Parliament", "type": "ORGANIZATION"}},
    {{"name": "AI Act", "type": "REGULATION"}},
    {{"name": "GPT-5", "type": "MODEL"}},
    {{"name": "Claude", "type": "MODEL"}},
    {{"name": "OpenAI", "type": "ORGANIZATION"}},
    {{"name": "Sam Altman", "type": "PERSON"}}
  ],
  "relations": [
    {{"subject": "EU Parliament", "predicate": "LAUNCHED", "object": "AI Act", "confidence": 0.95}},
    {{"subject": "AI Act", "predicate": "REGULATES", "object": "GPT-5", "confidence": 0.90}},
    {{"subject": "AI Act", "predicate": "REGULATES", "object": "Claude", "confidence": 0.90}},
    {{"subject": "Sam Altman", "predicate": "WORKS_AT", "object": "OpenAI", "confidence": 0.95}},
    {{"subject": "OpenAI", "predicate": "SUPPORTS", "object": "AI Act", "confidence": 0.85}}
  ]
}}

Text: "Anthropic raised $2B from Google, valuing the Claude maker at $18B. The startup competes with OpenAI and is headquartered in San Francisco."
Output:
{{
  "entities": [
    {{"name": "Anthropic", "type": "ORGANIZATION"}},
    {{"name": "Google", "type": "ORGANIZATION"}},
    {{"name": "Claude", "type": "MODEL"}},
    {{"name": "OpenAI", "type": "ORGANIZATION"}},
    {{"name": "San Francisco", "type": "LOCATION"}}
  ],
  "relations": [
    {{"subject": "Anthropic", "predicate": "FUNDED_BY", "object": "Google", "confidence": 0.95}},
    {{"subject": "Claude", "predicate": "CREATED_BY", "object": "Anthropic", "confidence": 0.90}},
    {{"subject": "Anthropic", "predicate": "COMPETES_WITH", "object": "OpenAI", "confidence": 0.90}},
    {{"subject": "Anthropic", "predicate": "HEADQUARTERED_IN", "object": "San Francisco", "confidence": 0.85}}
  ]
}}

NOW EXTRACT FROM THIS MESSAGE:
---
{message_content}
---

INSTRUCTIONS:
1. First pass: Extract all entities and relationships explicitly stated in the text.
2. Self-verification: Review each extracted triple. Verify it is EXPLICITLY stated or directly implied by the text. Remove any triple that requires inference beyond what the text states. Do not hallucinate relationships.
3. Use ONLY entity types and relation types from the menus above.
4. Assign confidence 0.0-1.0 based on how explicitly the relationship is stated (1.0 = directly stated, 0.7 = strongly implied, below 0.6 = do not include).

Return ONLY valid JSON:
{{
  "entities": [{{"name": "...", "type": "PERSON|ORGANIZATION|MODEL|TECHNOLOGY|PRODUCT|EVENT|REGULATION|LOCATION|CONCEPT|DATASET"}}],
  "relations": [{{"subject": "...", "predicate": "CREATED_BY|WORKS_AT|SUBSIDIARY_OF|HEADQUARTERED_IN|MEMBER_OF|FUNDED_BY|COMPETES_WITH|OUTPERFORMS|BASED_ON|SUCCESSOR_OF|ALTERNATIVE_TO|ANNOUNCED|ACQUIRED|LAUNCHED|PARTNERED_WITH|INVESTED_IN|SUPPORTS|OPPOSES|REGULATES|RELATED_TO", "object": "...", "confidence": 0.0}}]
}}
"""

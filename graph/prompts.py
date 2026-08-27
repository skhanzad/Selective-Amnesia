# ── Main Extraction Prompt ────────────────────────────────────────────────────

_MAIN_EXTRACTION_SYSTEM_PROMPT = """
You are a knowledge graph extraction engine. Given a single user conversational
message, extract all nodes, edges, and attributes and return them as strict JSON
conforming to the schema defined below. Extract all entities explicitly mentioned
or clearly implied, including objects, locations, participants, and evaluative language.

════════════════════════════════════════
CORE CONCEPTS
════════════════════════════════════════
NODE
  A discrete entity extracted from the message ie: a person, concept, skill, goal,
  event, artifact,emotional state, etc. A node can be a semantic node or episodic node
  explained below. Title of node should be general ie: red car -> car, classical guitar -> guitar
  Nodes are entities that may form relationships with other nodes.

EDGE
  A directed relationship between two nodes. Use precise, domain-relevant verbs. Avoid generic
  relationships like ‘has’, ‘related to’, or ‘about’ when a more specific relation is possible.

ATTRIBUTE
  A key-value property attached to a node or edge used to provide details. An
  attribute is qualified if it is unlikely to have more than 1 edge, is too detailed
  to be a node.

SEMANTIC NODE
  Represents a persistent fact that would remain true even if this conversation
  were forgotten (e.g. "User is a software engineer", "User plays guitar").

EPISODIC NODE
  Represents something that happened in or was requested during this specific
  interaction. It is time-bound and does not necessarily persist
  (e.g. "User went to the mall", "User felt frustrated").


════════════════════════════════════════
NODE TYPE ONTOLOGY  (use only these labels)
════════════════════════════════════════
  Person         — the user or people they reference (semantic)
  Belief         — opinions, values, or interpretations (semantic)
  Knowledge      — factual information the user understands (semantic)
  Preference     — expressed likes or dislikes (semantic)
  Skill          — things the user knows, is learning, or practises (semantic)
  Goal           — what the user is working toward (semantic)
  Trait          — inferred behavioural or personality patterns (semantic)
  Interest       — areas the user is drawn to (semantic)
  Topic          — subjects the user engages with repeatedly (semantic)
  Project        — a structured, multi-step effort (semantic)
  Artifact       — documents, code, or other outputs (semantic)
  Resource       — external materials used by the user (semantic)
  Responsibility — obligations the user is accountable for (semantic)

  Event          — something that happened (always episodic)
  State          — an ongoing condition or status (semantic)
  Habit          — recurring behaviour over time (semantic)
  Milestone      — a significant progress checkpoint (episodic)

  Location       — a physical or virtual place (semantic)
  Object         — a tangible item or possession (semantic)
  Organization   — a company, institution, or group (semantic)
  System         — an environment governed by rules or structure (semantic)
  Tool           — an instrument used to perform tasks (semantic)

  Role           — the user’s function in a context (semantic)

  Constraint     — a limitation or requirement (semantic)
  Metric         — a measure of success or performance (semantic)
  Risk           — a potential negative outcome (semantic)

  Query          — a question or request made in this turn (always episodic)
  Emotion        — inferred affective state during this session (always episodic)


════════════════════════════════════════
DATE HANDLING
════════════════════════════════════════
Do NOT attempt to resolve or normalise relative date expressions. When a value
contains a time expression (relative or absolute), output it verbatim in the
"raw_date" field and leave all resolution to the downstream date normalisation
module.

Examples:
  "last Saturday"    → "raw_date": "last Saturday"
  "a few months ago" → "raw_date": "a few months ago"
  "on June 1st"      → "raw_date": "on June 1st"
  "in 2019"          → "raw_date": "in 2019"
  
════════════════════════════════════════
OUTPUT SCHEMA
════════════════════════════════════════
Return ONLY valid JSON. No explanation, no markdown fences, no preamble.
The output must be a single JSON object matching this exact structure:

{
  "nodes": {
    <short snake_case identifier, e.g. node_0, node_1>: {
      "label"     : "<ontology label>",
      "title"     : "<short identifier ie: user, alice, mall trip>"
      "node_type" : "semantic" | "episodic",
      "content"   : "<clean third-person description>",
      "attributes": {
        "key1": { "value": "value1" },
        "key2": { "value": "value2" }
      }
    }
  },
  "edges": {
    <short snake_case identifier, e.g. edge_0, edge_1>: {
      "source"    : "<node id>",
      "target"    : "<node id>",
      "relationship" : "<relationship qualifier>",
      "attributes": {
        "key1": { "value": "value1" },
        "key2": { "value": "value2" }
      }
    }
  }
}

════════════════════════════════════════
FEW-SHOT EXAMPLES
════════════════════════════════════════

── Example 1: Physical descriptor + Contact detail ─────────────────────────
User message: "I'm 5'11, born in Lyon, France, and you can reach me at theo@example.com."

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person",
      "title": "user",
      "node_type": "semantic",
      "content": "The user",
      "attributes": {
        "height":     { "value": "5'11" },
        "birthplace": { "value": "Lyon, France" },
        "email":      { "value": "theo@example.com" }
      }
    }
  },
  "edges": {}
}

── Example 2: Skill + Preference ───────────────────────────────────────────
User message: "I've been playing classical guitar since high school and I really prefer fingerpicking over strumming."

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person",
      "title": "user",
      "node_type": "semantic",
      "content": "The user",
      "attributes": {
      }
    },
    "node_1": {
      "label": "Skill",
      "title": "guitar",
      "node_type": "semantic",
      "content": "User plays classical guitar",
      "attributes": {
      }
    },
    "node_2": {
      "label": "Preference",
      "title": "fingerpicking",
      "node_type": "semantic",
      "content": "User prefers fingerpicking over strumming",
      "attributes": {
      }
    }
  },
  "edges": {
    "edge_0": {
      "source": "node_0",
      "target": "node_1",
      "relationship": "plays",
      "attributes": {
        "qualifier": { "value": "classical" },
        "started":  { "value": "since high school" }
      }
    },
    "edge_1": {
      "source": "node_0",
      "target": "node_2",
      "relationship": "prefers",
      "attributes": {
      }
    }
  }
}

── Example 3: Relationship qualifier ───────────────────────────────────────
User message: "My sister Elena works at Google in London as a product manager."

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person",
      "title": "user",
      "node_type": "semantic",
      "content": "The user",
      "attributes": {
      }
    },
    "node_1": {
      "label": "Person",
      "title": "Elena",
      "node_type": "semantic",
      "content": "Elena, the user's sister",
      "attributes": {
        "role": { "value": "product manager" }
      }
    },
    "node_2": {
      "label": "Organization",
      "title": "Google",
      "node_type": "semantic",
      "content": "Google",
      "attributes": {
      }
    },
    "node_3": {
      "label": "Location",
      "title": "London",
      "node_type": "semantic",
      "content": "London",
      "attributes": {
      }
    }
  },
  "edges": {
    "edge_0": {
      "source": "node_0",
      "target": "node_1",
      "relationship": "related to",
      "attributes": {
        "type": { "value": "sister" }
      }
    },
    "edge_1": {
      "source": "node_1",
      "target": "node_2",
      "relationship": "works at",
      "attributes": {
      }
    },
    "edge_2": {
      "source": "node_1",
      "target": "node_3",
      "relationship": "works in",
      "attributes": {
      }
    }
  }
}

── Example 4: Artifact property ────────────────────────────────────────────
User message: "I wrote a Python CLI tool last year called 'grip' for automating my photo backups."

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person",
      "title": "user",
      "node_type": "semantic",
      "content": "The user",
      "attributes": {}
    },
    "node_1": {
      "label": "Artifact",
      "title": "grip",
      "node_type": "semantic",
      "content": "CLI tool 'grip' for automating photo backups",
      "attributes": {
        "name":     { "value": "grip" },
        "language": { "value": "Python" },
        "purpose":  { "value": "automating photo backups" },
        "raw_date": { "value": "last year" }
      }
    }
  },
  "edges": {
    "edge_0": {
      "source": "node_0",
      "target": "node_1",
      "relationship": "created",
      "attributes": {
        "raw_date": { "value": "last year" }
      }
    }
  }
}

── Example 5: Event ────────────────────────────────────────────────────────
User message: "I attended a machine learning conference last Saturday in Berlin."

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person",
      "title": "user",
      "node_type": "semantic",
      "content": "The user",
      "attributes": {
      }
    },
    "node_1": {
      "label": "Event",
      "title": "Machine Learning Conference Berlin",
      "node_type": "episodic",
      "content": "User attended a machine learning conference in Berlin",
      "attributes": {
        "raw_date": { "value": "last Saturday" }
      }
    },
    "node_2": {
      "label": "Location",
      "title": "Berlin",
      "node_type": "semantic",
      "content": "Berlin",
      "attributes": {
      }
    },
    "node_3": {
      "label": "Topic",
      "title": "machine learning",
      "node_type": "semantic",
      "content": "Machine learning",
      "attributes": {
      }
    }
  },
  "edges": {
    "edge_0": {
      "source": "node_0",
      "target": "node_1",
      "relationship": "attended",
      "attributes": {
        "raw_date": { "value": "last Saturday" }
      }
    },
    "edge_1": {
      "source": "node_1",
      "target": "node_2",
      "relationship": "located at",
      "attributes": {
      }
    },
    "edge_2": {
      "source": "node_1",
      "target": "node_3",
      "relationship": "about",
      "attributes": {
      }
    }
  }
}

── Example 6: Query (queryable intent) ─────────────────────────────────────
User message: "Can you explain how transformer attention mechanisms work?"

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person",
      "title": "user",
      "node_type": "semantic",
      "content": "The user",
      "attributes": {
      }
    },
    "node_1": {
      "label": "Query",
      "title": "transformer attention query",
      "node_type": "episodic",
      "content": "User requested an explanation of transformer attention mechanisms",
      "attributes": {
      }
    },
    "node_2": {
      "label": "Topic",
      "title": "transformer attention mechanisms",
      "node_type": "semantic",
      "content": "Transformer attention mechanisms",
      "attributes": {
      }
    }
  },
  "edges": {
    "edge_0": {
      "source": "node_0",
      "target": "node_1",
      "relationship": "made query",
      "attributes": {
      }
    },
    "edge_1": {
      "source": "node_1",
      "target": "node_2",
      "relationship": "about",
      "attributes": {
      }
    }
  }
}

── Example 7: Both informational and queryable ──────────────────────────────
User message: "I'm a nurse and I've been struggling with documenting patient care efficiently — do you have any tools or templates I could use?"

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person",
      "title": "user",
      "node_type": "semantic",
      "content": "The user",
      "attributes": {
        "occupation": { "value": "nurse" }
      }
    },
    "node_1": {
      "label": "Goal",
      "title": "efficient patient care documentation",
      "node_type": "semantic",
      "content": "User wants to document patient care more efficiently",
      "attributes": {}
    },
    "node_2": {
      "label": "Query",
      "title": "documentation tools query",
      "node_type": "episodic",
      "content": "User requested tools or templates for patient care documentation",
      "attributes": {}
    },
    "node_3": {
      "label": "Topic",
      "title": "patient care documentation",
      "node_type": "semantic",
      "content": "Patient care documentation",
      "attributes": {}
    }
  },
  "edges": {
    "edge_0": {
      "source": "node_0",
      "target": "node_1",
      "relationship": "has goal",
      "attributes": {
      }
    },
    "edge_1": {
      "source": "node_0",
      "target": "node_2",
      "relationship": "made query",
      "attributes": {
      }
    },
    "edge_2": {
      "source": "node_2",
      "target": "node_3",
      "relationship": "about",
      "attributes": {
      }
    },
    "edge_3": {
      "source": "node_0",
      "target": "node_3",
      "relationship": "engaged with",
      "attributes": {
      }
    }
  }
}

════════════════════════════════════════
USER MESSAGE TO EXTRACT FROM:
════════════════════════════════════════
"""

# ──  Extraction Prompt ───────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """
Extract a knowledge graph from a conversation turn. Return strict JSON only — no markdown, no preamble.

Each turn is prefixed with its speaker role:
  [Role: user]      — the human speaking directly
  [Role: assistant] — the AI assistant responding

ROLE HANDLING:
  User turns: extract facts about the user exactly as stated. "I went to Paris" → User visited Paris.
  Assistant turns: extract TWO categories of content:
    (a) User-related facts the assistant references or confirms.
        • Treat "you" as the user. "You mentioned you love hiking" → User loves hiking.
        • If the assistant states a fact about the user ("You've been using Python for 6 years"), extract it as a user fact.
    (b) Specific factual claims, recommendations, named entities, and concrete information the assistant states.
        • These capture what was said in the conversation so it can be recalled later.
        • Examples: algorithm names and their properties, recipe recommendations, budget figures, study statistics, suggested names or options, tool capabilities.
        • Connect these entities to each other or to the user's context as appropriate.
    Skip ONLY: pure pleasantries, generic filler, and assistant self-references with no content ("That's great!", "I can help with that", "Happy to assist!").
    Do NOT create Query nodes for rhetorical questions the assistant asks.

NODES: Discrete entities. Titles must be general and reusable (red car→car, classical guitar→guitar). content describes what the entity IS, never its relationship to the user.
EDGES: Directed relationships between nodes. Use specific verbs, not generic ones like "has" or "related to".
ATTRIBUTES: Captured facts as key/value pairs on edges by default. Only put an attribute on a node if it is intrinsic to that node alone (e.g. email, phone on Person).
  Dates: output verbatim in raw_date. e.g. "last Friday" → {"raw_date": "last Friday"}

RULES — these are hard constraints:
  1. Every node must connect to at least one other node via an edge. A node with no edges is invalid — store it as an attribute on a relevant edge instead.
  2. Every entity, relationship, and fact stated in the message must appear somewhere in the output. Nothing may be silently dropped.
  3. If an assistant turn is purely pleasantry with zero substantive content, return {"nodes": {}, "edges": {}}. A turn with any named entity, number, recommendation, or factual claim is NOT pure pleasantry.

LABELS — use only these nine:
  Person       — the user or any person referenced
  Organization — companies, institutions, groups
  Location     — physical or virtual places
  Event        — time-bound occurrences, interactions, emotions
  Concept      — abstract ideas: beliefs, topics, lists, habits, states, roles, metrics
  Artifact     — physical or tangible things: objects, properties, products, documents, tools, systems
  Preference   — an expressed like, dislike, or stated way of doing something (e.g. window seats, oat milk, dark mode)
  Goal         — something the user is working toward or wants to achieve
  Skill        — something the user knows how to do, is learning, or practises

SCHEMA:
{
  "nodes": {
    "<id>": {"label": "<label>", "title": "<short>", "content": "<what it is>", "attributes": {"key": "val"}}
  },
  "edges": {
    "<id>": {"source": "<node_id>", "target": "<node_id>", "relationship": "<verb>", "attributes": {"key": "val"}}
  }
}

── Example 1 (user turn) ───────────────────────────────────
[Role: user]
I go to the gym about 4 times a week and I have a playlist with 45 songs on it.

Output:
{
  "nodes": {
    "node_0": {"label": "Person", "title": "user", "content": "The user", "attributes": {}},
    "node_1": {"label": "Location", "title": "gym", "content": "Gym or fitness centre", "attributes": {}},
    "node_2": {"label": "Artifact", "title": "playlist", "content": "Music playlist", "attributes": {}}
  },
  "edges": {
    "edge_0": {"source": "node_0", "target": "node_1", "relationship": "attends", "attributes": {"frequency": "4 times a week"}},
    "edge_1": {"source": "node_0", "target": "node_2", "relationship": "maintains", "attributes": {"count": "45"}}
  }
}

── Example 2 (user turn) ───────────────────────────────────
[Role: user]
I bought a used Honda Civic last month for $12,000, financed through my local credit union.

Output:
{
  "nodes": {
    "node_0": {"label": "Person", "title": "user", "content": "The user", "attributes": {}},
    "node_1": {"label": "Artifact", "title": "car", "content": "Motor vehicle", "attributes": {}},
    "node_2": {"label": "Organization", "title": "credit union", "content": "Credit union", "attributes": {}}
  },
  "edges": {
    "edge_0": {"source": "node_0", "target": "node_1", "relationship": "purchased", "attributes": {"price": "$12,000", "model": "Honda Civic", "condition": "used", "raw_date": "last month"}},
    "edge_1": {"source": "node_0", "target": "node_2", "relationship": "financed through", "attributes": {"purpose": "car purchase"}}
  }
}

── Example 3 (user turn) ───────────────────────────────────
[Role: user]
My brother Tom just started a new job at Amazon in Seattle as a software engineer.

Output:
{
  "nodes": {
    "node_0": {"label": "Person", "title": "user", "content": "The user", "attributes": {}},
    "node_1": {"label": "Person", "title": "Tom", "content": "Tom, the user's brother", "attributes": {}},
    "node_2": {"label": "Organization", "title": "Amazon", "content": "Amazon", "attributes": {}},
    "node_3": {"label": "Location", "title": "Seattle", "content": "Seattle, Washington", "attributes": {}}
  },
  "edges": {
    "edge_0": {"source": "node_0", "target": "node_1", "relationship": "sibling of", "attributes": {}},
    "edge_1": {"source": "node_1", "target": "node_2", "relationship": "works at", "attributes": {"role": "software engineer", "tenure": "new"}},
    "edge_2": {"source": "node_2", "target": "node_3", "relationship": "located in", "attributes": {}}
  }
}

── Example 4 (user turn, complex) ──────────────────────────
[Role: user]
I'm a nurse, reach me at sarah@example.com. I've been learning Spanish for about 3 years and I'm at a B1 level. I went to a medical conference last Friday in Paris with my colleague James, who I've worked with for 5 years. My reading list has 15 books on it.

Output:
{
  "nodes": {
    "node_0": {"label": "Person", "title": "user", "content": "The user, a nurse", "attributes": {"email": "sarah@example.com", "occupation": "nurse"}},
    "node_1": {"label": "Skill", "title": "Spanish", "content": "Spanish language", "attributes": {}},
    "node_2": {"label": "Event", "title": "medical conference Paris", "content": "Medical conference held in Paris", "attributes": {"raw_date": "last Friday"}},
    "node_3": {"label": "Location", "title": "Paris", "content": "Paris, France", "attributes": {}},
    "node_4": {"label": "Person", "title": "James", "content": "James, user's colleague", "attributes": {}},
    "node_5": {"label": "Concept", "title": "reading list", "content": "A curated list of books to read", "attributes": {}}
  },
  "edges": {
    "edge_0": {"source": "node_0", "target": "node_1", "relationship": "learning", "attributes": {"duration": "about 3 years", "level": "B1"}},
    "edge_1": {"source": "node_0", "target": "node_2", "relationship": "attended", "attributes": {"raw_date": "last Friday"}},
    "edge_2": {"source": "node_2", "target": "node_3", "relationship": "located at", "attributes": {}},
    "edge_3": {"source": "node_2", "target": "node_4", "relationship": "attended by", "attributes": {}},
    "edge_4": {"source": "node_0", "target": "node_4", "relationship": "works with", "attributes": {"duration": "5 years"}},
    "edge_5": {"source": "node_0", "target": "node_5", "relationship": "maintains", "attributes": {"count": "15"}}
  }
}

── Example 5 (user turn) ───────────────────────────────────
[Role: user]
I always book window seats when I fly and I hate aisle seats. I'm trying to get fit this year — I want to run a 5K by summer. I've been doing Python professionally for 6 years.

Output:
{
  "nodes": {
    "node_0": {"label": "Person", "title": "user", "content": "The user", "attributes": {}},
    "node_1": {"label": "Preference", "title": "window seat", "content": "Preference for window seats on flights", "attributes": {}},
    "node_2": {"label": "Preference", "title": "aisle seat", "content": "Aversion to aisle seats on flights", "attributes": {}},
    "node_3": {"label": "Goal", "title": "run 5K", "content": "Goal to run a 5K by summer", "attributes": {"deadline": "summer"}},
    "node_4": {"label": "Skill", "title": "Python", "content": "Python programming language", "attributes": {}}
  },
  "edges": {
    "edge_0": {"source": "node_0", "target": "node_1", "relationship": "prefers", "attributes": {"strength": "always", "context": "flights"}},
    "edge_1": {"source": "node_0", "target": "node_2", "relationship": "dislikes", "attributes": {"context": "flights"}},
    "edge_2": {"source": "node_0", "target": "node_3", "relationship": "working toward", "attributes": {}},
    "edge_3": {"source": "node_0", "target": "node_4", "relationship": "practises", "attributes": {"experience_years": "6", "context": "professional"}}
  }
}

── Example 6 (assistant turn — confirming user facts) ──────
[Role: assistant]
That's impressive! So you've been doing Python professionally for 6 years and you're currently working toward running a 5K by summer. Based on what you've told me, you always book window seats when flying. Would you like some training plan recommendations?

Output:
{
  "nodes": {
    "node_0": {"label": "Person", "title": "user", "content": "The user", "attributes": {}},
    "node_1": {"label": "Skill", "title": "Python", "content": "Python programming language", "attributes": {}},
    "node_2": {"label": "Goal", "title": "run 5K", "content": "Goal to run a 5K by summer", "attributes": {"deadline": "summer"}},
    "node_3": {"label": "Preference", "title": "window seat", "content": "Preference for window seats on flights", "attributes": {}}
  },
  "edges": {
    "edge_0": {"source": "node_0", "target": "node_1", "relationship": "practises", "attributes": {"experience_years": "6", "context": "professional"}},
    "edge_1": {"source": "node_0", "target": "node_2", "relationship": "working toward", "attributes": {}},
    "edge_2": {"source": "node_0", "target": "node_3", "relationship": "prefers", "attributes": {"context": "flights"}}
  }
}

── Example 7 (assistant turn — pure pleasantry, no facts) ──
[Role: assistant]
That sounds great! I'd be happy to help you with that. Could you tell me a bit more about what you're looking for?

Output:
{"nodes": {}, "edges": {}}

── Example 8 (assistant turn — technical recommendation with named entities) ──
[Role: assistant]
SIAC_GEE and Sen2Cor are both tools for atmospheric correction of Sentinel-2 images. SIAC_GEE uses the 6S radiative transfer model, while Sen2Cor uses the L2A_Process algorithm. SIAC_GEE runs on Google Earth Engine and is open-source; Sen2Cor is a standalone commercial tool.

Output:
{
  "nodes": {
    "node_0": {"label": "Artifact", "title": "SIAC_GEE", "content": "Atmospheric correction tool for Sentinel-2 images running on Google Earth Engine", "attributes": {"platform": "Google Earth Engine", "license": "open-source"}},
    "node_1": {"label": "Artifact", "title": "Sen2Cor", "content": "Standalone atmospheric correction tool for Sentinel-2 images", "attributes": {"license": "commercial"}},
    "node_2": {"label": "Concept", "title": "6S radiative transfer model", "content": "Algorithm for atmospheric correction", "attributes": {}},
    "node_3": {"label": "Concept", "title": "L2A_Process algorithm", "content": "Algorithm for atmospheric correction used by Sen2Cor", "attributes": {}}
  },
  "edges": {
    "edge_0": {"source": "node_0", "target": "node_2", "relationship": "uses", "attributes": {}},
    "edge_1": {"source": "node_1", "target": "node_3", "relationship": "uses", "attributes": {}},
    "edge_2": {"source": "node_0", "target": "node_1", "relationship": "compared to", "attributes": {}}
  }
}

Conversation turn to extract from:
"""

EXHAUSTIVE_EXTRACTION_SYSTEM_PROMPT = """
You are a knowledge graph extraction engine. Given a single user conversational
message, extract all nodes, edges, and attributes and return them as strict JSON
conforming to the schema defined below.

════════════════════════════════════════
CORE CONCEPTS
════════════════════════════════════════
NODE
  A discrete entity extracted from the message: a person, concept, skill, goal,
  event, artifact, emotional state, etc. A node can be semantic or episodic (see below).
  Keep node titles general so they can be reused across contexts:
    red car → car,  classical guitar → guitar,  to-watch list → to-watch list

EDGE
  A directed relationship between two nodes. Use precise, domain-relevant verbs.
  Avoid generic relationships like 'has', 'related_to', or 'about' when a more
  specific relation is possible.

ATTRIBUTE
  A key-value property that captures a stated fact. Attributes belong on edges by
  default. Put an attribute on a node only if it is intrinsic to that node and
  could not meaningfully belong to any relationship (e.g. a phone number or email
  address on a Person node).

  WHY EDGES: Nodes are kept general so they can be shared. If "user has 20 titles
  on their to-watch list", the count 20 is a fact about the user's relationship to
  the list, not about the list itself. Storing it on the edge preserves the node
  for reuse by other entities.

  EXHAUSTIVENESS RULE: Every fact the user explicitly states must appear in the
  output — as a node title, edge relationship, or edge/node attribute. No stated
  fact may be silently dropped. This includes:
    • numeric quantities and counts  ("20 titles" → edge attribute count: 20)
    • amounts and costs              ("spent $200" → edge attribute spent: 200)
    • sizes, durations, frequencies  ("twice a week" → edge attribute frequency: "twice a week")
    • stated states or conditions    ("currently injured" → node State or edge attribute)
  If a fact cannot be expressed as a node or edge, add it as an attribute on the
  most relevant edge. Do not discard it.

SEMANTIC NODE
  A persistent fact that would remain true even if this conversation were forgotten
  (e.g. "User is a software engineer", "User plays guitar").

EPISODIC NODE
  Something that happened or was requested in this specific interaction. Time-bound;
  does not necessarily persist (e.g. "User went to the mall", "User felt frustrated").


════════════════════════════════════════
NODE LABEL ONTOLOGY  (use only these labels)
════════════════════════════════════════
  Person         — the user or people they reference (semantic)
  Belief         — opinions, values, or interpretations (semantic)
  Knowledge      — factual information the user understands (semantic)
  Preference     — expressed likes or dislikes (semantic)
  Skill          — things the user knows, is learning, or practises (semantic)
  Goal           — what the user is working toward (semantic)
  Trait          — inferred behavioural or personality patterns (semantic)
  Interest       — areas the user is drawn to (semantic)
  Topic          — subjects the user engages with repeatedly (semantic)
  Project        — a structured, multi-step effort (semantic)
  Artifact       — documents, code, or other outputs (semantic)
  Resource       — external materials used by the user (semantic)
  Responsibility — obligations the user is accountable for (semantic)

  Event          — something that happened (always episodic)
  State          — an ongoing condition or status (semantic)
  Habit          — recurring behaviour over time (semantic)
  Milestone      — a significant progress checkpoint (episodic)

  Location       — a physical or virtual place (semantic)
  Object         — a tangible item or possession (semantic)
  Organization   — a company, institution, or group (semantic)
  System         — an environment governed by rules or structure (semantic)
  Tool           — an instrument used to perform tasks (semantic)

  Role           — the user's function in a context (semantic)

  Constraint     — a limitation or requirement (semantic)
  Metric         — a measure of success or performance (semantic)
  Risk           — a potential negative outcome (semantic)

  Query          — a question or request made in this turn (always episodic)
  Emotion        — inferred affective state during this session (always episodic)


════════════════════════════════════════
DATE HANDLING
════════════════════════════════════════
Do NOT attempt to resolve or normalise relative date expressions. When a value
contains a time expression (relative or absolute), output it verbatim in the
"raw_date" field and leave resolution to the downstream date normalisation module.

  "last Saturday"    → "raw_date": "last Saturday"
  "a few months ago" → "raw_date": "a few months ago"
  "on June 1st"      → "raw_date": "on June 1st"
  "in 2019"          → "raw_date": "in 2019"

════════════════════════════════════════
OUTPUT SCHEMA
════════════════════════════════════════
Return ONLY valid JSON. No explanation, no markdown fences, no preamble.

{
  "nodes": {
    <short snake_case id, e.g. node_0>: {
      "label"    : "<ontology label>",
      "title"    : "<short identifier>",
      "node_type": "semantic" | "episodic",
      "content"  : "<what this entity IS, not its relationship to the user>",
      "attributes": {
        "<key>": { "value": "<val>" | false }
      }
    }
  },
  "edges": {
    <short snake_case id, e.g. edge_0>: {
      "source"      : "<node id>",
      "target"      : "<node id>",
      "relationship": "<relationship verb>",
      "attributes"  : {
        "<key>": { "value": "<val>" | false }
      }
    }
  }
}

════════════════════════════════════════
FEW-SHOT EXAMPLES
════════════════════════════════════════

── Example 1: Simple ───────────────────────────────────────────────────────
User message: "I've got a to-watch list with 20 titles on it."

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person", "title": "user", "node_type": "semantic",
      "content": "The user", "attributes": {}
    },
    "node_1": {
      "label": "Topic", "title": "to-watch list", "node_type": "semantic",
      "content": "A curated list of titles to watch", "attributes": {}
    }
  },
  "edges": {
    "edge_0": {
      "source": "node_0", "target": "node_1", "relationship": "maintains",
      "attributes": {
        "count": { "value": 20 }
      }
    }
  }
}

── Example 2: Complex ──────────────────────────────────────────────────────
User message: "I'm a nurse, reach me at sarah@example.com. I've been learning Spanish for about 3 years and I'm at a B1 level. I went to a medical conference last Friday in Paris with my colleague James, who I've worked with for 5 years. My reading list has 15 books on it — can you recommend something to add?"

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person", "title": "user", "node_type": "semantic",
      "content": "The user, a nurse",
      "attributes": {
        "email":      { "value": "sarah@example.com" },
        "occupation": { "value": "nurse" }
      }
    },
    "node_1": {
      "label": "Skill", "title": "Spanish", "node_type": "semantic",
      "content": "Spanish language", "attributes": {}
    },
    "node_2": {
      "label": "Event", "title": "medical conference Paris", "node_type": "episodic",
      "content": "Medical conference held in Paris",
      "attributes": { "raw_date": { "value": "last Friday" } }
    },
    "node_3": {
      "label": "Location", "title": "Paris", "node_type": "semantic",
      "content": "Paris, France", "attributes": {}
    },
    "node_4": {
      "label": "Person", "title": "James", "node_type": "semantic",
      "content": "James", "attributes": {}
    },
    "node_5": {
      "label": "Topic", "title": "reading list", "node_type": "semantic",
      "content": "A curated list of books to read", "attributes": {}
    },
    "node_6": {
      "label": "Query", "title": "book recommendation query", "node_type": "episodic",
      "content": "Request for a book recommendation", "attributes": {}
    }
  },
  "edges": {
    "edge_0": {
      "source": "node_0", "target": "node_1", "relationship": "learning",
      "attributes": {
        "duration": { "value": "about 3 years" },
        "level":    { "value": "B1" }
      }
    },
    "edge_1": {
      "source": "node_0", "target": "node_2", "relationship": "attended",
      "attributes": { "raw_date": { "value": "last Friday" } }
    },
    "edge_2": {
      "source": "node_2", "target": "node_3", "relationship": "located at",
      "attributes": {}
    },
    "edge_3": {
      "source": "node_2", "target": "node_4", "relationship": "attended by",
      "attributes": {}
    },
    "edge_4": {
      "source": "node_0", "target": "node_4", "relationship": "works with",
      "attributes": { "duration": { "value": "5 years" } }
    },
    "edge_5": {
      "source": "node_0", "target": "node_5", "relationship": "maintains",
      "attributes": { "count": { "value": 15 } }
    },
    "edge_6": {
      "source": "node_0", "target": "node_6", "relationship": "made query",
      "attributes": {}
    },
    "edge_7": {
      "source": "node_6", "target": "node_5", "relationship": "about",
      "attributes": {}
    }
  }
}

════════════════════════════════════════
USER MESSAGE TO EXTRACT FROM:
════════════════════════════════════════
"""

# ── Retrieval Extraction Prompt ────────────────────────────────────────────────

RETRIEVAL_SYSTEM_PROMPT = """
Extract entities from this question for knowledge graph lookup. Return strict JSON only — no markdown, no preamble.

Goal: identify what the question is asking about so matching nodes can be found in a personal knowledge graph. Focus on the subjects being asked about, not the question itself.

LABELS — use only these nine:
  Person       — the user or any person referenced
  Organization — companies, institutions, groups
  Location     — physical or virtual places
  Event        — time-bound occurrences, interactions, emotions
  Concept      — abstract ideas: beliefs, topics, lists, habits, states, roles, metrics
  Artifact     — physical or tangible things: objects, properties, products, documents, tools, systems
  Preference   — an expressed like, dislike, or stated way of doing something
  Goal         — something the user is working toward or wants to achieve
  Skill        — something the user knows how to do, is learning, or practises

SCHEMA:
{"entities": [{"label": "<label>", "title": "<short reusable title>", "content": "<what it is>"}]}

EXAMPLES:
Q: "Where does my mom live?"
{"entities": [{"label": "Person", "title": "mom", "content": "User's mother"}]}

Q: "What is the name of my sister's dog?"
{"entities": [{"label": "Person", "title": "sister", "content": "User's sister"}, {"label": "Artifact", "title": "dog", "content": "Pet dog"}]}

Q: "How many years have I been playing chess?"
{"entities": [{"label": "Skill", "title": "chess", "content": "Chess game"}]}

Q: "What programming languages does Alice know?"
{"entities": [{"label": "Person", "title": "Alice", "content": "Person named Alice"}, {"label": "Skill", "title": "programming languages", "content": "Programming language skills"}]}

Q: "What kind of coffee do I like?"
{"entities": [{"label": "Preference", "title": "coffee", "content": "Coffee preference"}]}

Q: "What are my fitness goals?"
{"entities": [{"label": "Goal", "title": "fitness", "content": "Fitness or exercise goal"}]}

Question:
"""

# ── Trimmed Extraction Prompt ──────────────────────────────────────────────────

TRIMMED_EXTRACTION_SYSTEM_PROMPT = """
You are a knowledge graph extraction engine. Given a single user conversational
message, extract all nodes, edges, and attributes and return them as strict JSON
conforming to the schema defined below. Extract all entities explicitly mentioned
or clearly implied, including objects, locations, participants, and evaluative language.

════════════════════════════════════════
CORE CONCEPTS
════════════════════════════════════════
NODE
  A discrete entity extracted from the message ie: a person, concept, skill, goal,
  event, artifact, emotional state, etc. A node can be a semantic node or episodic node
  explained below. Title of node should be general ie: red car -> car, classical guitar -> guitar
  Nodes are entities that may form relationships with other nodes.

EDGE
  A directed relationship between two nodes. Use precise, domain-relevant verbs. Avoid generic
  relationships like 'has', 'related to', or 'about' when a more specific relation is possible.

ATTRIBUTE
  A key-value property attached to a node or edge used to provide details. An
  attribute is qualified if it is unlikely to have more than 1 edge, is too detailed
  to be a node.

SEMANTIC NODE
  Represents a persistent fact that would remain true even if this conversation
  were forgotten (e.g. "User is a software engineer", "User plays guitar").

EPISODIC NODE
  Represents something that happened in or was requested during this specific
  interaction. It is time-bound and does not necessarily persist
  (e.g. "User went to the mall", "User felt frustrated").


════════════════════════════════════════
NODE TYPE ONTOLOGY  (use only these labels)
════════════════════════════════════════
  Person         — the user or people they reference (semantic)
  Belief         — opinions, values, or interpretations (semantic)
  Knowledge      — factual information the user understands (semantic)
  Preference     — expressed likes or dislikes (semantic)
  Skill          — things the user knows, is learning, or practises (semantic)
  Goal           — what the user is working toward (semantic)
  Trait          — inferred behavioural or personality patterns (semantic)
  Interest       — areas the user is drawn to (semantic)
  Topic          — subjects the user engages with repeatedly (semantic)
  Project        — a structured, multi-step effort (semantic)
  Artifact       — documents, code, or other outputs (semantic)
  Resource       — external materials used by the user (semantic)
  Responsibility — obligations the user is accountable for (semantic)

  Event          — something that happened (always episodic)
  State          — an ongoing condition or status (semantic)
  Habit          — recurring behaviour over time (semantic)
  Milestone      — a significant progress checkpoint (episodic)

  Location       — a physical or virtual place (semantic)
  Object         — a tangible item or possession (semantic)
  Organization   — a company, institution, or group (semantic)
  System         — an environment governed by rules or structure (semantic)
  Tool           — an instrument used to perform tasks (semantic)

  Role           — the user's function in a context (semantic)

  Constraint     — a limitation or requirement (semantic)
  Metric         — a measure of success or performance (semantic)
  Risk           — a potential negative outcome (semantic)

  Query          — a question or request made in this turn (always episodic)
  Emotion        — inferred affective state during this session (always episodic)

════════════════════════════════════════
DATE HANDLING
════════════════════════════════════════
Do NOT attempt to resolve or normalise relative date expressions. When a value
contains a time expression (relative or absolute), output it verbatim in the
"raw_date" field and leave all resolution to the downstream date normalisation
module.

Examples:
  "last Saturday"    → "raw_date": "last Saturday"
  "a few months ago" → "raw_date": "a few months ago"
  "on June 1st"      → "raw_date": "on June 1st"
  "in 2019"          → "raw_date": "in 2019"

════════════════════════════════════════
OUTPUT SCHEMA
════════════════════════════════════════
Return ONLY valid JSON. No explanation, no markdown fences, no preamble.
The output must be a single JSON object matching this exact structure:

{
  "nodes": {
    <short snake_case identifier, e.g. node_0, node_1>: {
      "label"     : "<ontology label>",
      "title"     : "<short identifier ie: user, alice, mall trip>"
      "node_type" : "semantic" | "episodic",
      "content"   : "<clean third-person description>",
      "attributes": {
        "key1": { "value": "value1" },
        "key2": { "value": "value2" }
      }
    }
  },
  "edges": {
    <short snake_case identifier, e.g. edge_0, edge_1>: {
      "source"       : "<node id>",
      "target"       : "<node id>",
      "relationship" : "<relationship qualifier>",
      "attributes"   : {
        "key1": { "value": "value1" },
        "key2": { "value": "value2" }
      }
    }
  }
}

════════════════════════════════════════
FEW-SHOT EXAMPLES
════════════════════════════════════════

── Example 1: Skill + Preference ───────────────────────────────────────────
User message: "I've been playing classical guitar since high school and I really prefer fingerpicking over strumming."

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person",
      "title": "user",
      "node_type": "semantic",
      "content": "The user",
      "attributes": {}
    },
    "node_1": {
      "label": "Skill",
      "title": "guitar",
      "node_type": "semantic",
      "content": "User plays classical guitar",
      "attributes": {}
    },
    "node_2": {
      "label": "Preference",
      "title": "fingerpicking",
      "node_type": "semantic",
      "content": "User prefers fingerpicking over strumming",
      "attributes": {}
    }
  },
  "edges": {
    "edge_0": {
      "source": "node_0",
      "target": "node_1",
      "relationship": "plays",
      "attributes": {
        "qualifier": { "value": "classical" },
        "started":   { "value": "since high school" }
      }
    },
    "edge_1": {
      "source": "node_0",
      "target": "node_2",
      "relationship": "prefers",
      "attributes": {}
    }
  }
}

── Example 2: Event (episodic node + location) ──────────────────────────────
User message: "I attended a machine learning conference last Saturday in Berlin."

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person",
      "title": "user",
      "node_type": "semantic",
      "content": "The user",
      "attributes": {}
    },
    "node_1": {
      "label": "Event",
      "title": "Machine Learning Conference Berlin",
      "node_type": "episodic",
      "content": "User attended a machine learning conference in Berlin",
      "attributes": {
        "raw_date": { "value": "last Saturday" }
      }
    },
    "node_2": {
      "label": "Location",
      "title": "Berlin",
      "node_type": "semantic",
      "content": "Berlin",
      "attributes": {}
    },
    "node_3": {
      "label": "Topic",
      "title": "machine learning",
      "node_type": "semantic",
      "content": "Machine learning",
      "attributes": {}
    }
  },
  "edges": {
    "edge_0": {
      "source": "node_0",
      "target": "node_1",
      "relationship": "attended",
      "attributes": {
        "raw_date": { "value": "last Saturday" }
      }
    },
    "edge_1": {
      "source": "node_1",
      "target": "node_2",
      "relationship": "located at",
      "attributes": {}
    },
    "edge_2": {
      "source": "node_1",
      "target": "node_3",
      "relationship": "about",
      "attributes": {}
    }
  }
}

── Example 3: Both informational and queryable ──────────────────────────────
User message: "I'm a nurse and I've been struggling with documenting patient care efficiently — do you have any tools or templates I could use?"

Output:
{
  "nodes": {
    "node_0": {
      "label": "Person",
      "title": "user",
      "node_type": "semantic",
      "content": "The user",
      "attributes": {
        "occupation": { "value": "nurse" }
      }
    },
    "node_1": {
      "label": "Goal",
      "title": "efficient patient care documentation",
      "node_type": "semantic",
      "content": "User wants to document patient care more efficiently",
      "attributes": {}
    },
    "node_2": {
      "label": "Query",
      "title": "documentation tools query",
      "node_type": "episodic",
      "content": "User requested tools or templates for patient care documentation",
      "attributes": {}
    },
    "node_3": {
      "label": "Topic",
      "title": "patient care documentation",
      "node_type": "semantic",
      "content": "Patient care documentation",
      "attributes": {}
    }
  },
  "edges": {
    "edge_0": {
      "source": "node_0",
      "target": "node_1",
      "relationship": "has goal",
      "attributes": {}
    },
    "edge_1": {
      "source": "node_0",
      "target": "node_2",
      "relationship": "made query",
      "attributes": {}
    },
    "edge_2": {
      "source": "node_2",
      "target": "node_3",
      "relationship": "about",
      "attributes": {}
    },
    "edge_3": {
      "source": "node_0",
      "target": "node_3",
      "relationship": "engaged with",
      "attributes": {}
    }
  }
}

════════════════════════════════════════
USER MESSAGE TO EXTRACT FROM:
════════════════════════════════════════
"""

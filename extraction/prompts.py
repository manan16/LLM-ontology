from __future__ import annotations

from ingestion.chunking import DocumentChunk


SYSTEM_PROMPT = """
You are a compliance-aware ontology extraction engine for HIPAA and EU AI Act legal text.
Extract duties, permissions, prohibitions, controls, risks, rights, documentation duties, monitoring duties, and source citations only when explicitly supported by the source chunk.
Do not infer missing actors, obligations, conditions, exceptions, risks, deadlines, safeguards, or citations.
If a field is not present, return null or an empty list.
Return only valid JSON matching the provided schema. Do not add markdown.
Use deterministic wording and keep canonical names concise.
Preserve exact supporting evidence text and citation metadata.
""".strip()


def build_extraction_prompt(chunk: DocumentChunk) -> str:
    return f"""
Task:
Extract compliance ontology statements, entities, and relationships from this HIPAA or EU AI Act text chunk.

Rules:
- Return ONLY schema-valid JSON.
- Extract only information explicitly present in the chunk.
- If a field is absent, use null or [].
- Copy evidence_text exactly from the chunk.
- Preserve citation/source text metadata such as Article, Section, Recital, Annex, clause, page, or paragraph references when present.
- Do not invent actors, legal bases, exceptions, controls, safeguards, monitoring duties, deadlines, or risks.
- Keep canonical_name concise and normalized; keep raw_text literal.

For each compliance rule, identify:
- who has the duty
- what action is required, allowed, or prohibited
- what data, system, process, policy, procedure, or record the rule applies to
- conditions
- exceptions
- deadlines
- documentation or record-keeping requirements
- reporting or notification requirements
- human oversight requirements
- monitoring requirements
- safeguards and controls
- risks
- affected rights
- citation/source text

Classification guidance:
- "must", "shall", "required to", "obligated to" means obligation or requirement.
- "must not", "shall not", "prohibited", "forbidden" means prohibition.
- "may", "permitted", "allowed" means permission.
- Extract exceptions and conditions separately from the main obligation.
- Preserve exact supporting evidence text and citation metadata.

HIPAA focus:
- Covered Entity, Business Associate, Protected Health Information (PHI), authorization, use/disclosure, minimum necessary rule, individual rights, privacy notice, safeguards, training, complaints, sanctions, policies, procedures, documentation, accounting of disclosures.

EU AI Act focus:
- Provider, Deployer, AI System, High-Risk AI System, risk management, human oversight, technical documentation, logs, post-market monitoring, incident reporting, conformity assessment, fundamental rights impact assessment.

Useful relationship types include:
- IMPOSES_ON, REQUIRES_AUTHORIZATION, REQUIRES_DOCUMENTATION, REQUIRES_ASSESSMENT, REQUIRES_MONITORING, REQUIRES_HUMAN_OVERSIGHT, HAS_CONDITION, HAS_DEADLINE, HAS_SAFEGUARD, HAS_CONTROL, PROTECTS_RIGHT, MITIGATES_RISK, REPORTS_TO, NOTIFIES, DOCUMENTED_BY, LIMITED_BY, DISCLOSES_TO, USES_FOR, HAS_PURPOSE, DERIVED_FROM, CITES.

Chunk metadata:
- chunk_id: {chunk.chunk_id}
- section_id: {chunk.section_id}
- section_title: {chunk.section_title}
- chunk_index: {chunk.chunk_index}
- start_char: {chunk.start_char}
- end_char: {chunk.end_char}

Source chunk:
{chunk.text}
""".strip()

# AI Defence Lab
## Adaptive Attack Family Design

**Version:** Adaptive design v1  
**Purpose:** Define how the final red-team/blue-team loop should choose and improve attack families.

## 1. The Core Idea

The system should use a **hybrid family policy**:

- A stable approved taxonomy provides repeatable benchmark categories.
- Detector weaknesses drive the next attack family.
- Agent 1 recommends the next direction using evidence and Attack Memory.
- Agent 2 converts that direction into structured constraints.
- The generator creates valid synthetic transactions.
- No new families are created; all rounds remain within the approved taxonomy.

The base documents do not need to change every round. They provide general payment-security knowledge. The experiment memory provides what the current detector is failing to recognize.

## 2. The Approved Family Set

### Approved benchmark families

These are stable categories used for comparable experiments:

- account_takeover
- trusted_device
- beneficiary_manipulation
- low_and_slow
- social_engineering
- merchant_abuse
- cross_channel_anomaly

Each family has a definition, observable features, generator support, and evaluation rules. This is taxonomy version 1, not a claim that all possible attacks are known.

Agent 1 may change the feature-level direction within one of these families, but it may not introduce variants, composites, or discovery labels.

## 3. The Round-by-Round Flow

```text
Approved taxonomy gives initial candidate
                 |
                 v
Agent 1 reads base documents + Attack Memory
                 |
                 v
Agent 1 recommends a family or variant
                 |
                 v
Controller validates the recommendation
                 |
                 v
Agent 2 writes structured constraints
                 |
                 v
Generator creates synthetic rows
                 |
                 v
Detector evaluates unseen rows
                 |
                 v
Agent 3 records the weakness in Attack Memory
                 |
                 +--------> next round
```

The controller remains responsible for the final decision. Agent 1 is adaptive, but it cannot introduce an invalid family or bypass the generator and schema checks.

## 4. Simple Example

### Round 1: account takeover

The seed selects `account_takeover` as the initial family. The detector evaluates unseen rows and reports:

> Many missed attacks used normal devices and did not show a device change.

Agent 3 stores this weakness in Attack Memory.

### Round 2: trusted device

Agent 1 reads the weakness and recommends:

```json
{
  "recommended_family": "trusted_device",
  "reason": "The detector relies too heavily on device novelty.",
        "recommended_family": "trusted_device"
}
```

Agent 2 creates a specification:

- device change near the legitimate baseline
- moderate beneficiary changes
- normal transaction velocity
- controlled amounts
- ordinary channels

The generator then creates rows from these constraints. The LLM does not directly invent raw transaction rows.

### Round 3: low and slow

Suppose the next weakness says:

> Trusted-device attacks are detected when velocity is elevated, but missed when activity is spread across time.

Agent 1 recommends `low_and_slow`. Agent 2 specifies modest amounts, low velocity, normal channels, and occasional beneficiary changes.

### Round 4: approved-family continuation

Suppose the detector still misses attacks that combine a familiar device with low velocity. Agent 1 continues with the approved `trusted_device` or `low_and_slow` family and changes the structured feature constraints; it does not create a composite family.

## 5. How Realism Is Protected

The adaptive recommendation changes the research direction, but several controls protect synthetic quality:

1. **Family profiles:** each family maps to realistic feature ranges and relationships.
2. **Agent 2 constraints:** the specification states what should remain normal and what should change.
3. **Schema validation:** amounts, hours, binary fields, velocity, channels, and labels are checked.
4. **Reference comparison:** generated rows are compared with legitimate synthetic reference data.
5. **Strict labels:** mixed-family generator output is rejected, not silently relabelled.
6. **Unseen evaluation:** detector test rows are separate from generator and detector training rows.
7. **Diversity checks:** duplicate ratio, channel coverage, numeric variation, and family coverage are recorded.

Therefore, the base documents provide knowledge, Agent 1 provides adaptation, Agent 2 provides structure, and the generator plus validators provide realistic data.

## 6. Final Decision Policy

The final family scheduler should follow this order:

1. Start with a seeded approved family so the experiment is reproducible.
2. Use Agent 3's latest weakness to guide the next recommendation.
3. Let Agent 1 propose only one of the seven approved families.
4. Validate the recommendation against the taxonomy and recent history.
5. Let Agent 2 create constraints from the recommendation and its parent profile.
6. Generate and validate synthetic rows.
7. Keep the taxonomy versioned while retaining the same seven-family allowlist.

## Final Architecture

```text
Stable base knowledge
        +
Seeded approved taxonomy
        +
Detector weaknesses in Attack Memory
        |
        v
Adaptive Agent 1 recommendation
        |
        v
Validated Agent 2 specification
        |
        v
Procedural, GAN, or diffusion generator
        |
        v
Schema, realism, diversity, and unseen evaluation checks
        |
        v
Agent 3 weakness report -> Attack Memory -> next round
```

The final design is neither completely fixed nor completely uncontrolled. It is a stable benchmark with evidence-driven adaptation and a controlled discovery process.

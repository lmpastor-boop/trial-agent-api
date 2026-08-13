# Clinical Trial Navigator: Portfolio Product Roadmap

## Product statement

Clinical Trial Navigator is a clinician-in-the-loop application that retrieves
recruiting studies from ClinicalTrials.gov, applies deterministic eligibility
checks, and uses an LLM to rank plausible matches with criterion-grounded
rationales.

It is a portfolio demonstration built with synthetic patient summaries. It is
not a medical device, a final eligibility determination, or a HIPAA-compliant
clinical system.

## Intended user and problem

**Primary user:** a clinical research coordinator performing an initial trial
screen before a physician or principal investigator makes the final decision.

**Current pain:** coordinators repeatedly search long eligibility documents and
manually compare criteria against incomplete patient information.

**Desired outcome:** reduce initial screening time while preserving traceability
to the source criteria and escalating uncertain cases to a human reviewer.

## Portfolio success criteria

The project is ready to headline an FDE application when it demonstrates:

1. A public, usable workflow backed only by synthetic or de-identified examples.
2. Automated tests and CI for deterministic behavior and API contracts.
3. A documented evaluation set of at least 50 cases, including difficult
   exclusions and missing-information cases.
4. Criterion-level citations or evidence spans for every model recommendation.
5. Observable latency, token cost, tool calls, failures, and fallback behavior.
6. Authentication, restricted CORS, rate limiting, and an explicit threat model.
7. A cloud deployment with managed Postgres, health checks, logs, and a
   repeatable deployment process.
8. A concise case study covering discovery, architecture, trade-offs,
   evaluation results, failure analysis, and measured workflow impact.

## Milestones

### Milestone 1 — Engineering baseline

- Convert the smoke script into an offline pytest suite.
- Run the suite automatically on every push and pull request.
- Add typed application settings and input constraints.
- Separate external ClinicalTrials.gov and model clients from graph logic.
- Add structured logging and correlation IDs.

**Exit evidence:** a green CI badge and reproducible local test command.

### Milestone 2 — Trustworthy matching

- Return structured criterion evidence with each verdict.
- Validate model output against a strict schema.
- Add timeouts, retry limits, and explicit fallback states.
- Expand the hand-labeled evaluation set to at least 50 cases.
- Report per-class precision/recall, abstention rate, faithfulness, latency,
  and estimated cost.
- Maintain a written failure taxonomy.

**Exit evidence:** an evaluation report generated from a versioned dataset.

### Milestone 3 — Usable reviewer workflow

- Build a small React/Next.js interface.
- Provide synthetic patient scenarios rather than accepting real PHI.
- Display source criteria beside model recommendations.
- Capture reviewer accept/reject decisions and reasons.
- Add a human-approval checkpoint before any downstream action.

**Exit evidence:** a deployed end-to-end demo that a reviewer can use without
Swagger or command-line tools.

### Milestone 4 — Deployment and operations

- Deploy the containerized API and frontend to AWS.
- Use managed Postgres and secure secret storage.
- Add authentication, authorization, rate limiting, restricted CORS, and audit
  events.
- Add CI/CD, health/readiness checks, dashboards, and alerts.
- Document rollback and incident-response procedures.

**Exit evidence:** a repeatable deployment plus an operations runbook.

### Milestone 5 — Healthcare integration story

- Accept a small synthetic FHIR Bundle containing Patient, Condition,
  Observation, and MedicationStatement resources.
- Map those fields into the matching representation.
- Document HIPAA, BAA, retention, access-control, and de-identification
  requirements that would apply to a real implementation.
- Conduct several structured user interviews or usability sessions.

**Exit evidence:** a short case study showing how field feedback changed the
product and what would be required for safe enterprise adoption.

## First four implementation tickets

1. **Automated API regression suite and CI** — establishes repeatable delivery.
2. **Typed settings and safe input boundaries** — removes configuration hidden
   in module imports and rejects unsuitable input early.
3. **Structured model response with evidence** — makes outputs testable and
   reviewable.
4. **Request tracing and operational metrics** — makes failures, latency, and
   cost visible.

## Skills demonstrated

Completing the roadmap provides evidence of the core FDE competencies:
customer discovery, ambiguous problem framing, Python/API engineering,
full-stack delivery, data integration, agent evaluation, production
operations, healthcare risk judgment, and measurable adoption.

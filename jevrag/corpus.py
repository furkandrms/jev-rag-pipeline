"""Synthetic internal engineering knowledge base for a fictional company, Nimbus.

Deliberately messier than a trivia corpus: overlapping topics (multiple
runbooks mention "restart the service" for different services), documents
that partially but not fully answer plausible questions, and a couple of
gaps (no on-call escalation doc past Slack, no data-retention policy) so
sufficiency gating has real work to do, not just an easy relevance filter.
Nimbus, its services, and every incident/policy below are fictional.
"""

from __future__ import annotations

DOCUMENTS: list[tuple[str, str]] = [
    # --- Onboarding ---------------------------------------------------------
    ("onboard-1", "New engineers at Nimbus get repo access via the #it-access Slack channel; approval usually takes under a day."),
    ("onboard-2", "Local dev setup: clone nimbus/platform, run `make bootstrap`, then `make dev` to start all services against a local Postgres."),
    ("onboard-3", "Every new engineer is paired with an onboarding buddy for their first two weeks who reviews their first three PRs."),
    ("onboard-4", "Nimbus's engineering handbook lives in nimbus/handbook; the architecture overview doc there is required reading in week one."),
    ("onboard-5", "VPN access requires a hardware security key, requested through IT during your first day of orientation."),
    # --- API reference: Ingest service --------------------------------------
    ("api-ingest-1", "POST /v1/ingest accepts a JSON batch of up to 500 events; larger batches return a 413."),
    ("api-ingest-2", "The ingest service enforces a rate limit of 1000 requests per minute per API key, returned as a 429 with a Retry-After header when exceeded."),
    ("api-ingest-3", "Ingest events must include a `tenant_id` field; requests missing it are rejected with a 400 before reaching the queue."),
    ("api-ingest-4", "Ingested events are asynchronously written to the events-raw Kafka topic, then compacted into events-daily every night at 02:00 UTC."),
    # --- API reference: Query service ---------------------------------------
    ("api-query-1", "GET /v1/query/{tenant_id} returns aggregated event counts for the last 30 days by default; use `?range=` to extend up to 1 year."),
    ("api-query-2", "The query service caches results per tenant for 5 minutes; pass `?fresh=true` to bypass the cache for debugging."),
    ("api-query-3", "Query service authentication uses the same API keys as ingest, but requires the `read:events` scope specifically."),
    # --- Runbooks: Ingest service --------------------------------------------
    ("runbook-ingest-1", "If the ingest service's queue depth alert fires, first check Kafka consumer lag in the ingest-consumers dashboard before restarting anything."),
    ("runbook-ingest-2", "To restart the ingest service in production, use `kubectl rollout restart deployment/ingest -n prod`; this is a rolling restart, so no downtime is expected."),
    ("runbook-ingest-3", "A sustained 413 error spike from ingest usually means an upstream client started sending oversized batches; check the client's recent deploys before assuming a Nimbus-side bug."),
    # --- Runbooks: Query service ----------------------------------------------
    ("runbook-query-1", "To restart the query service in production, use `kubectl rollout restart deployment/query -n prod`; unlike ingest, this briefly drops the read-through cache and can cause a short latency spike."),
    ("runbook-query-2", "If query latency p99 exceeds 2s, check whether the Postgres read replica lag alert is also firing -- they're almost always correlated."),
    ("runbook-query-3", "Query service memory leaks have historically been traced to unbounded result-set caching; check `query_cache_size_bytes` before restarting as a first response."),
    # --- Runbooks: general on-call --------------------------------------------
    ("runbook-oncall-1", "The on-call engineer is paged via PagerDuty; the current rotation is visible in the #oncall Slack channel topic."),
    ("runbook-oncall-2", "Sev1 incidents require posting in #incidents within 5 minutes of acknowledgment and opening a postmortem doc from the template."),
    ("runbook-oncall-3", "On-call handoff happens every Monday at 10:00 UTC; the outgoing on-call must summarize open issues in the #oncall channel before handing off."),
    # --- Incident postmortems --------------------------------------------------
    ("incident-1", "2025-11-03 incident: the ingest service ran out of database connections after a config change raised max batch size without raising the connection pool limit; fix was reverting the batch size change and adding a pool-size alert."),
    ("incident-2", "2025-12-14 incident: query service p99 latency spiked to 8s for 40 minutes due to a missing index after a schema migration; root cause was the migration checklist not including an index-verification step."),
    ("incident-3", "2026-01-22 incident: ingest silently dropped events for one tenant for 3 hours because a malformed tenant_id passed validation but failed downstream partitioning; fix added stricter tenant_id format validation at the API boundary."),
    ("incident-4", "2026-02-09 incident: on-call was paged but didn't see the alert for 25 minutes due to a PagerDuty routing misconfiguration after a team rename; fix was adding a routing-config check to the team-rename checklist."),
    # --- Policies ---------------------------------------------------------------
    ("policy-deploy-1", "All production deploys require at least one approving code review and must pass CI, including the integration test suite, before merging."),
    ("policy-deploy-2", "Deploys are only allowed Monday through Thursday before 16:00 UTC, except for hotfixes explicitly approved by the on-call lead."),
    ("policy-review-1", "PRs touching the ingest or query service's public API require review from a member of the API working group, not just any engineer."),
    ("policy-review-2", "Nimbus does not have a strict PR size limit, but reviewers are encouraged to ask for a split if a diff exceeds roughly 400 lines."),
    ("policy-security-1", "Any API key suspected of being leaked must be revoked immediately via the admin console and reported in #security within one hour."),
    ("policy-security-2", "Nimbus rotates all internal service credentials automatically every 90 days; manual rotation is only needed after a suspected compromise."),
    # --- Misc / FAQ ---------------------------------------------------------------
    ("faq-1", "Nimbus's staging environment mirrors production topology but runs on a 10x smaller Kafka cluster, so load testing there is not representative of production throughput."),
    ("faq-2", "The #platform-questions Slack channel is the right place for \"how do I...\" questions about ingest or query; #incidents is only for active incidents."),
    ("faq-3", "Nimbus does not currently support batch deletion of events via the API; deletion requests go through a manual process handled by the data team."),
    ("faq-4", "The events-daily Kafka topic is retained for 400 days; events-raw is retained for only 7 days since it's just a staging area before compaction."),
]

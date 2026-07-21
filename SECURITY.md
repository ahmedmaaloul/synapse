# Security Policy

Thanks for helping keep Synapse and its users safe. Security reports are
genuinely welcome, and researchers who report responsibly will be credited.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report privately to:

**ahmed.maaloul@proton.me** (Ahmed Maaloul)

Use the subject line `[SECURITY] Synapse — <short summary>`. If you
prefer, you may instead use GitHub's private
[Security Advisories](https://github.com/ahmedmaaloul/synapse/security/advisories/new)
flow on the canonical repository.

### What to include

The more of this you can provide, the faster a fix lands:

- A description of the issue and the impact you believe it has
- The affected component (backend API, graph/Cypher layer, retrieval pipeline,
  frontend, Docker/deployment config) and the version, tag, or commit SHA
- Step-by-step reproduction instructions, or a proof-of-concept
- Any relevant logs, requests, or configuration (with secrets redacted)
- Whether the issue is already public or under a third-party disclosure deadline

## Response Targets

| Stage | Target |
| --- | --- |
| Acknowledgement of your report | within **72 hours** |
| Initial assessment and severity triage | within **7 days** |
| Fix or documented mitigation for high/critical issues | within **30 days** |
| Fix for low/medium issues | next scheduled release, or within 90 days |

This is a personally maintained open source project, so these are good-faith
targets rather than a contractual SLA. If you have not heard back within 72
hours, please send a follow-up — mail does occasionally go astray.

## Responsible Disclosure

We ask that you give us **90 days** from the date of your report before
disclosing publicly, so a fix and an advisory can be prepared. If a fix ships
sooner, we will coordinate a disclosure date with you and are happy to publish
earlier. If we fail to respond or to remediate within that window, you are free
to disclose.

During that period, please:

- Test only against your own installation, never against someone else's
  deployment or data
- Avoid privacy violations, data destruction, and service degradation
- Avoid social engineering, phishing, and physical attacks

Research conducted in good faith under this policy will not be pursued or
reported by the maintainer as a violation of any applicable law or of this
project's terms.

Please note there is no bug bounty program and no monetary reward for this
project. Credit in the security advisory and release notes is offered for every
valid report, unless you prefer to remain anonymous.

## Supported Versions

Security fixes are applied to the latest release and to the `main` branch. Older
minor versions are not backported.

| Version | Supported |
| --- | --- |
| `main` (unreleased) | Yes |
| 1.x (latest minor) | Yes |
| 1.x (older minors) | No — please upgrade |
| < 1.0 (pre-release) | No |

If you run a fork or a modified deployment, you are responsible for applying
upstream security fixes to your own build.

## Scope

**In scope:** the code in this repository — the FastAPI backend, the Neo4j graph
and Cypher layer, the retrieval and AI-provider abstraction, the Next.js
frontend, and the shipped Docker and deployment configuration.

**Out of scope:**

- Vulnerabilities in third-party dependencies — please report those upstream,
  though a heads-up here is appreciated so the pin can be bumped
- Vulnerabilities in Neo4j, model providers, or other external services
- Issues that require an already-compromised host or an already-leaked
  credential
- Insecure configuration that the documentation explicitly warns against, such
  as running the development compose stack with default credentials on a public
  network
- Missing hardening headers or best practices with no demonstrated impact, and
  automated-scanner output submitted without a working proof-of-concept

## Security Expectations for Operators

Synapse is provided **without warranty** under the AGPL-3.0-or-later
license (see `LICENSE`). If you self-host it, you are responsible for your own
deployment: change all default credentials, keep secrets and API keys out of
version control, do not expose Neo4j or the backend API directly to the public
internet without authentication and TLS, and keep your dependencies current.

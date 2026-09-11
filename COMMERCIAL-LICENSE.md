# Commercial Licensing

Synapse is **source-available under the PolyForm Noncommercial License 1.0.0**,
with a commercial license for everyone that license does not cover. You get to
pick the option that fits what you are building:

| | Noncommercial (free) | Commercial (paid) |
| --- | --- | --- |
| License | [PolyForm-Noncommercial-1.0.0](LICENSE) | Private agreement |
| Cost | Free | Negotiated |
| Who it is for | Individuals, researchers, students, educators, charities and the other nonprofits the license names, government | Any company or for-profit purpose — internal tools included, on-prem or SaaS |
| Use inside a business | No | Yes |
| Ship inside a product or hosted service | Noncommercial ones only | Yes |
| Keep your modifications private | Yes — there is no copyleft | Yes |
| Warranty, indemnity, support | None | Negotiable |
| Attribution to Ahmed Maaloul | Required — the `Required Notice:` line | Required — the `Required Notice:` line plus a credits-screen mention |

Anyone learning from, researching with, or tinkering on Synapse lands in the
left column and never needs to talk to anyone. The commercial license exists
for one case, and it is a broad one: **you are using Synapse for a business.**

> **Not legal advice.** This page is a plain-English summary written for
> convenience. It is *not* legal advice and it is *not* the license. The
> [`LICENSE`](LICENSE) file is the binding document; where this page and
> `LICENSE` disagree, `LICENSE` wins. If the stakes are real, talk to a lawyer.

---

## The free option: PolyForm Noncommercial 1.0.0

For a **noncommercial purpose**, at no cost and without asking permission, you
may:

- **Self-host it** — on your laptop, your lab's cluster, your university's or
  your nonprofit's infrastructure.
- **Study it** — read every line, take it apart, learn from it, write about it.
- **Modify it** — change anything you want, for any reason, and keep the
  changes to yourself if you like. There is no copyleft.
- **Fork it** — publicly, on GitHub or anywhere else.
- **Redistribute it** — copies and modified versions alike, as long as the
  license terms and the `Required Notice:` line travel with them.
- **Contribute back** — pull requests are very welcome.

"Noncommercial" is defined by the license, not by this page. In short it
covers:

- **Personal use** — research, experiment and testing for the benefit of public
  knowledge, personal study, private entertainment, hobby projects, amateur
  pursuits or religious observance, *without any anticipated commercial
  application*.
- **Noncommercial organisations** — charities, educational institutions, public
  research organisations, public safety or health organisations, environmental
  protection organisations and government institutions, regardless of how they
  are funded.

In exchange for all of the above you must:

1. **Keep the notices.** Anyone who gets a copy of any part of Synapse from you
   must also get the license terms (or their URL) and the `Required Notice:`
   line at the top of [`LICENSE`](LICENSE). Keeping `LICENSE` and
   [`NOTICE`](NOTICE) intact does exactly that.
2. **Stay noncommercial.** The moment a use is commercial, the free license
   stops covering it — see immediately below, this is the part that catches
   people out.

### What "commercial" means here

It is worth being blunt:

> **Under PolyForm Noncommercial, use by or for a business is commercial — even
> if it is purely internal, and even if you never redistribute anything.**

Some concrete consequences:

- **Internal tools count.** A company running Synapse on its own servers, for
  its own employees, over its own documents, is using it commercially.
- **Redistribution is irrelevant.** The copyleft-era question — "do I have to
  publish my changes?" — has no counterpart here. The question is *who is using
  it and why*, never *who receives a copy*.
- **Hosting counts.** Offering Synapse, modified or not, as a service for or to
  a business is commercial.
- **"Free of charge" is not "noncommercial".** An internal proof of concept
  that costs nobody anything is still use for the benefit of a business.
- **The purpose is what matters, not the person.** A student is noncommercial;
  the same person building the same thing at their employer's request is not.

If you are unsure, the license's own definitions — "Noncommercial Purposes",
"Personal Uses" and "Noncommercial Organizations" — are short and in plain
English; read them in [`LICENSE`](LICENSE). If you are still unsure after
that, ask. The answer is usually quick.

### The client package needs no license at all

The `synapse-graphrag` package in
[`packages/synapse-graphrag/`](packages/synapse-graphrag/) — the MCP server,
the CLI and the Python SDK — is licensed under **Apache-2.0**, with its own
[`LICENSE`](packages/synapse-graphrag/LICENSE) and
[`NOTICE`](packages/synapse-graphrag/NOTICE). Embed it in anything, commercial
or not, with no obligations beyond Apache's. What it talks to — a Synapse
backend — is what the noncommercial-or-commercial choice is about.

---

## When you need a commercial license

Get in touch if any of these describe you:

- **You use it at a company.** Any for-profit organisation, for any purpose —
  in production or internally, on-prem or as SaaS, modified or vanilla.
- **You ship it inside a product.** Synapse (or a derivative) embedded in
  software or a service you sell, license, or otherwise put in front of
  customers.
- **You host it for others** as a paid or otherwise commercial service.
- **You build it for a client** as a contractor or consultancy — the client's
  use is what matters (see the FAQ).
- **You want a warranty, indemnity, or support commitment.** The noncommercial
  license explicitly provides none of these. A commercial agreement can.

A commercial license covers commercial use. It does **not** waive attribution —
credit to Ahmed Maaloul is required under either option, and a credits-screen
attribution (see [`NOTICE`](NOTICE)) is a term of the commercial agreement.

**Only Ahmed Maaloul can grant a commercial license.** No fork, redistributor,
or third party has that authority.

---

## How to get one

Email **Ahmed Maaloul** — <ahmed.maaloul@proton.me> — with subject line
`[Commercial License] <your company>`.

Helpful things to include, so the first reply can be useful:

- Who you are and what you are building
- How Synapse fits in — internal tool, embedded in a product, hosted for
  customers?
- Rough scale (users, deployments, seats) and your timeline
- Whether you need support, a warranty, or indemnification
- Whether you want an evaluation license first (see below)

Terms are negotiated case by case. Startups, small teams and academic
spin-outs should say so — pricing is flexible, and the goal is a workable
arrangement, not a toll booth. **Evaluation licenses** — time-boxed, free or
nominal, for a proof of concept inside a company — are available on request.

---

## FAQ

**Can I evaluate it at my company before deciding?**
Ask for an evaluation license — it is a short email, and the answer is normally
yes. Strictly, a proof of concept inside a company is use for the benefit of a
business, which the noncommercial license does not cover, so please do not rely
on "we are only testing it". Evaluating it *personally* — on your own machine,
in your own time, to learn how it works — needs nothing from anyone.

**I am a contractor or agency building on Synapse for a client. Who needs the
license?**
If the client is a business, or the work has a commercial purpose, the use is
commercial. Usually the client takes the commercial license, since they are the
ones using it; a contractor can also hold one that covers work for named
clients. Either way, get in touch before the engagement rather than after.

**Can I host Synapse for other people?**
For or to a business, or for money: that is commercial, and you need the
commercial license. A nonprofit or a university hosting it for its own members
for a noncommercial purpose is covered by the free license.

**Can I use it internally at my company?**
That needs the commercial license. Internal use by a business is commercial
under this license even though nothing leaves your network. This is the biggest
change from the AGPL-licensed versions, where internal use was free; it is
deliberate, and pricing for internal-only use reflects it.

**Do I need a license to self-host for my own personal use?**
No. Personal noncommercial use requires nothing from you beyond keeping the
license and notices in place. There is no registration, no key, and nothing to
pay.

**Can I fork it and contribute?**
Please do — that is the point of publishing the source. Fork it, open issues,
send pull requests. Your fork stays under the same license and keeps the
`Required Notice:` line, `LICENSE` and `NOTICE` intact (see
[`NOTICE`](NOTICE)); we ask that you mark your changes as yours. Contributions
come with the [Contributor License Agreement](CLA.md), which is what lets
contributed code be offered under both licenses.

**Does the noncommercial license apply to the MCP server, CLI and SDK too?**
No — that package is Apache-2.0. See
[above](#the-client-package-needs-no-license-at-all).

**What about the third-party dependencies?**
They keep their own licenses (see `backend/requirements.txt` and
`frontend/package.json`). A commercial license from Ahmed Maaloul covers
Ahmed's code in this repository; it cannot and does not change the terms of
anyone else's software.

**I bought a commercial license. Do I still have to credit the author?**
Yes. Attribution is required under both options: keep the `Required Notice:`
line, `LICENSE` and `NOTICE` in every copy, and — as a term of the commercial
license — add a credits or "About" screen attribution in the form
[`NOTICE`](NOTICE) describes.

**What about versions released before this change?**
They stay as published. Everything up to and including commit `91ee2f2` was
released under MIT and remains MIT. Versions 0.3.0 and 0.4.0 — every commit
after `91ee2f2` up to and including the `v0.4.0` tag (`34300d6`) — were
released under AGPL-3.0-or-later and remain available under it, copyleft
obligations included. The PolyForm Noncommercial + commercial model applies
from the change that introduced it onward (0.5.0 and later); code from 0.5.0 on
is not available under the AGPL.

**Can my contribution be included in the commercially licensed version?**
Yes, and that is worth being upfront about: for this model to work, the
maintainer needs the right to offer contributed code under the commercial
license too. By ticking the CLA checkbox in the pull-request template you agree
that your contribution may be shipped under the PolyForm Noncommercial License
**and** the commercial terms described here — and, for the client package,
under Apache-2.0. You keep the copyright in your own contribution. The full
text is short: [`CLA.md`](CLA.md).

---

Copyright (c) 2026 Ahmed Maaloul · <https://github.com/ahmedmaaloul/synapse> ·
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 (core) · Apache-2.0
(`packages/synapse-graphrag`)

# Commercial Licensing

Synapse is **dual-licensed**. You get to pick the option that fits what
you are building:

| | Open source | Commercial |
| --- | --- | --- |
| License | AGPL-3.0-or-later | Private agreement |
| Cost | Free, forever | Negotiated |
| Copyleft obligations | Yes | Waived |
| Must publish your modifications | Yes, if you serve them over a network | No |
| Ship inside a closed-source product | No | Yes |
| Attribution to Ahmed Maaloul | Required | Required |

Almost everyone lands in the left column and never needs to talk to anyone.
The commercial license exists for one narrow case: you want to build something
proprietary on top of Synapse and not release the source.

> **Not legal advice.** This page is a plain-English summary written for
> convenience. It is *not* legal advice and it is *not* the license. The
> [`LICENSE`](LICENSE) file is the binding document; where this page and
> `LICENSE` disagree, `LICENSE` wins. If the stakes are real, talk to a lawyer.

---

## The free option: AGPL-3.0-or-later

Under the AGPL you may, at no cost and without asking permission:

- **Self-host it** — for yourself, for your team, for your company, in
  production, including for commercial purposes and to make money.
- **Study it** — read every line, take it apart, learn from it, write about it.
- **Modify it** — change anything you want, for any reason.
- **Fork it** — publicly, on GitHub or anywhere else.
- **Redistribute it** — including selling copies or paid hosting.
- **Contribute back** — pull requests are very welcome.
- **Use it internally** — run it on your company's infrastructure for your own
  operations.

A common misconception worth clearing up: **the AGPL does not ban commercial
use.** You can absolutely use Synapse to make money. What the AGPL asks in
return is *reciprocity* — if you distribute it or serve it over a network, the
people on the other end get the same freedoms you got.

In exchange for all of the above you must:

1. **Keep the license.** Derivative works and modified versions must also be
   AGPL-3.0-or-later. You cannot relicense it under more restrictive terms.
2. **Keep the attribution.** Retain the copyright notices, `LICENSE`, and
   `NOTICE`. See [`NOTICE`](NOTICE) for exactly what this means, including
   user-facing credits screens.
3. **Provide source when you convey it.** Anyone you give a binary or a
   deployment to is entitled to the corresponding source.
4. **Honour section 13.** See immediately below — this is the clause that makes
   the AGPL different from the plain GPL, and the one that catches people out.

### AGPL section 13: the network/SaaS clause

This is the heart of the AGPL, so it is worth quoting the obligation plainly:

> **If you modify Synapse and let users interact with your modified
> version remotely over a network, you must prominently offer those users an
> opportunity to receive the complete corresponding source code of your modified
> version, at no charge.**

Two details that matter, in both directions:

- **It is triggered by *use over a network*, not by distribution.** The ordinary
  GPL only kicks in when you hand someone software. The AGPL closes that gap:
  running modified code as a hosted service counts, even though nobody ever
  downloads a copy. "Just SaaS-ing it" is not a way around copyleft here.
- **It is triggered by *modification*.** If you deploy Synapse completely
  unmodified, section 13 adds no new obligation of its own — there is no
  modified version whose source you would need to offer. In practice, though,
  anyone running a real deployment ends up changing *something*, so plan for
  this clause rather than betting against it.

"Complete corresponding source" means what it says: the whole modified work —
your changes to the backend, the frontend, the retrieval pipeline, build
scripts, everything needed to build and run it — not a diff or a summary.

---

## When you need a commercial license

Get in touch if any of these describe you:

- **Closed-source product.** You want to embed Synapse (or a derivative) in
  software you ship without publishing the source.
- **Proprietary SaaS.** You want to run a modified Synapse as a hosted service
  for your customers without offering them the corresponding source under
  section 13.
- **Copyleft is a blocker.** Your legal or procurement team prohibits AGPL
  dependencies, or you need to combine Synapse with proprietary code whose
  license is incompatible with the AGPL.
- **Relicensing.** You want to redistribute Synapse or a derivative under terms
  other than AGPL-3.0-or-later.
- **You want a warranty, indemnity, or support commitment.** The AGPL
  explicitly provides none of these. A commercial agreement can.

A commercial license waives the copyleft obligations for you. It does **not**
waive attribution — credit to Ahmed Maaloul is required under either option.

**Only Ahmed Maaloul can grant a commercial license.** No fork, redistributor,
or third party has that authority.

---

## How to get one

Email **Ahmed Maaloul** — <ahmed.maaloul@proton.me> — with subject line
`[Commercial License] <your company>`.

Helpful things to include, so the first reply can be useful:

- Who you are and what you are building
- How Synapse fits in — embedded, hosted for customers, internal-only?
- Rough scale (users, deployments, seats) and your timeline
- Whether you need support, a warranty, or indemnification

Terms are negotiated case by case. Startups, small teams, academic spin-outs,
and nonprofits should say so — pricing is flexible, and the goal is a workable
arrangement, not a toll booth.

---

## FAQ

**Can I evaluate it before deciding?**
Yes. Download it, run it, benchmark it, build a prototype, show it to your
team — no permission, no license purchase, no time limit. The AGPL covers
evaluation like any other use. You only need to think about the commercial
license when you go to ship something proprietary.

**Can I use it internally at my company?**
Yes. Running Synapse on your own infrastructure for your own operations is
fully permitted under the AGPL, free of charge, including at a for-profit
company. The one thing to be aware of: if you modify it and your employees use
it over your internal network, section 13 means those employees are entitled to
the corresponding source. Since they are inside your organisation, satisfying
that is usually as simple as pointing them at your internal Git repository. No
obligation to publish anything externally arises from internal use.

**Can I fork it and contribute?**
Please do — that is the point of publishing it. Fork it, open issues, send pull
requests. Your fork must stay under AGPL-3.0-or-later and keep the attribution
intact (see [`NOTICE`](NOTICE)), and you should mark your changes as yours, but
otherwise it is yours to take wherever you want.

**Do I need a license to self-host for my own personal use?**
No. Self-hosting for yourself requires nothing from you beyond keeping the
license and notices in place. There is no registration, no key, and nothing to
pay.

**I want to offer Synapse as a paid hosted service. Is that allowed?**
Under the AGPL, yes — selling hosting is explicitly permitted. The condition is
that if you have modified it, your users must be offered the corresponding
source of your modified version. If you would rather keep your modifications
private, that is exactly what the commercial license is for.

**Does the AGPL "infect" the rest of my codebase?**
It applies to Synapse and to works derived from it. Whether a separate program
that merely talks to Synapse over HTTP forms a single combined work is a
genuinely fact-specific legal question, and this document cannot answer it for
your architecture. If your integration is close enough that you are unsure, ask
your counsel — or take the commercial license and stop worrying about it.

**What about the third-party dependencies?**
They keep their own licenses (see `backend/requirements.txt` and
`frontend/package.json`). A commercial license from Ahmed Maaloul covers
Ahmed's code in this repository; it cannot and does not change the terms of
anyone else's software.

**I bought a commercial license. Do I still have to credit the author?**
Yes. Attribution is required under both options. See [`NOTICE`](NOTICE) for the
specific wording and where it needs to appear.

**Can my contribution be included in the commercially licensed version?**
Yes, and that is worth being upfront about: for a dual-licensing model to work,
the maintainer needs the right to offer contributed code under the commercial
license too. By submitting a pull request you agree that your contribution is
licensed under AGPL-3.0-or-later **and** that Ahmed Maaloul may also license it
under the commercial terms described here. You keep the copyright in your own
contribution. If a formal CLA is ever required, it will be added to the
repository and you will be asked to sign it explicitly.

---

Copyright (c) 2026 Ahmed Maaloul · <https://github.com/ahmedmaaloul/synapse> ·
SPDX-License-Identifier: AGPL-3.0-or-later

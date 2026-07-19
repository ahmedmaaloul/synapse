# Retrieval Evaluation Results

Hybrid GraphRAG retrieval (vector + full-text seeds → graph expansion) on the fixture in `eval/dataset.json`. Queries are paraphrased to avoid lexical overlap with entity names, so keyword-only search would miss most of them.

| Metric | Score |
| --- | --- |
| Hit@1 | 88% |
| Recall@8 | 100% |
| Precision@8 | 17% |
| MRR | 0.917 |

<details><summary>Per-query breakdown</summary>

| Question | Expected | Retrieved (top-5) | Recall |
| --- | --- | --- | --- |
| Which neural network libraries does the engineer use? | PyTorch, TensorFlow | scikit-learn, Ahmed Maaloul, PyTorch, TensorFlow, React | 100% |
| What technology stores data as connected nodes and edges? | Neo4j | Neo4j, AWS, Kubernetes, Knowledge Graphs, PostgreSQL | 100% |
| How does the team scale and orchestrate its containers? | Kubernetes, Docker | Kubernetes, Ahmed Maaloul, Docker, TechCorp, AWS | 100% |
| Where is the cloud infrastructure hosted? | AWS | AWS, Kubernetes, Docker, FastAPI, TechCorp | 100% |
| What methods ground LLMs in external facts? | Retrieval-Augmented Generation, Knowledge Graphs | Retrieval-Augmented Generation, PostgreSQL, scikit-learn, Neo4j, Knowledge Graphs | 100% |
| Where did Ahmed get his engineering degree? | ESILV | ESILV, Ahmed Maaloul, TechCorp, FastAPI, scikit-learn | 100% |
| What library builds web user interfaces? | React | React, FastAPI, Docker, scikit-learn, Kubernetes | 100% |
| Which relational SQL store is used? | PostgreSQL | PostgreSQL, Neo4j, AWS, Knowledge Graphs, Retrieval-Augmented Generation | 100% |

</details>

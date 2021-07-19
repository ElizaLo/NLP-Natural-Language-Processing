# Knowledge Graph

## Knowledge Graph vs Ontology

### What is the difference?

A knowledge base (KB) is fact-oriented but ontology is schema-oriented.

KB: In the Google KGraph, you have certainly a schema (like the one described by DBPedia, Freebase, etc.) and a set of facts (A relation B, "Paris isA City, Paris hasInhabitants 2M, Paris isCapitalOf France, France isA Country, etc.). So, we can (easily) answer to questions like "Number of inhabitants in Paris?" or you can provide a short description of Paris, France (or any given entity in general).

The domain ontology tells people which are the main concepts of a domain, how are these concepts related and which attributes do they have. Here the focus is on the description, with the highest possible expressiveness (disjointness, values restrictions, cardinality restrictions) and the useful annotations (synonyms, definition of terms, examples, comments on design choices, etc.), of the entities of a given domain. Data (or facts) are not the main concern when designing a domain ontology VS KB.

## What is the difference between a Knowledge Graph and a Graph Database?

Knowledge graphs are data. They have to be stored, managed, extended, quality-assured and can be queried. This requires databases and components on top, which are usually implemented in the Semantic Middleware Layer. This ‘sits’ on the database and at the same time offers service endpoints for integration with third-party systems.

Thus graph databases form the foundation of every knowledge graph. Typically, these are technologies based either on the **Resource Description Framework (RDF)**, a W3C standard, or on Labeled Property Graphs (LPG).

In order to roll out knowledge graphs in companies, however, more than a database is required: Only with the help of components such as taxonomy and ontology editors, entity extractors, graph mappers, validation, visualization and search tools, etc. can it be ensured that a knowledge graph can be sustainably developed and managed. While graph databases are typically maintained by highly qualified data engineers or Semantic Web experts, the interfaces of the Semantic Middleware also allow people to interact with the knowledge graph who can contribute less technical knowledge instead of business and expert knowledge to the graphs.

## Graph Databases

- [Neo4j](https://neo4j.com/?_gl=1%2A16rmv03%2A_ga%2AOTM3OTY0NzAxLjE2MjY2OTk4Nzc.%2A_ga_DL38Q8KGQC%2AMTYyNjY5OTg3Ni4xLjEuMTYyNjcwMDA4My4w&_ga=2.37309350.780889525.1626699877-937964701.1626699877)
  - > Neo4j gives developers and data scientists the most trusted and advanced tools to quickly build today’s intelligent applications and machine learning workflows. Available as a fully managed cloud service or self-hosted.
- 

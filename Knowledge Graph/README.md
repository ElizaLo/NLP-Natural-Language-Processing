# Knowledge Graph

## Knowledge Graph vs Ontology

### What is the difference?

A knowledge base (KB) is fact-oriented but ontology is schema-oriented.

KB: In the Google KGraph, you have certainly a schema (like the one described by DBPedia, Freebase, etc.) and a set of facts (A relation B, "Paris isA City, Paris hasInhabitants 2M, Paris isCapitalOf France, France isA Country, etc.). So, we can (easily) answer to questions like "Number of inhabitants in Paris?" or you can provide a short description of Paris, France (or any given entity in general).

The domain ontology tells people which are the main concepts of a domain, how are these concepts related and which attributes do they have. Here the focus is on the description, with the highest possible expressiveness (disjointness, values restrictions, cardinality restrictions) and the useful annotations (synonyms, definition of terms, examples, comments on design choices, etc.), of the entities of a given domain. Data (or facts) are not the main concern when designing a domain ontology VS KB.

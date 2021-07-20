# Ontology

- [Newest 'ontology' Questions](https://stackoverflow.com/questions/tagged/ontology)

## Aticles

- [Ontology and Data Science](https://towardsdatascience.com/ontology-and-data-science-45e916288cc5)

## Papers

### Food Industry

- [FoodOn: a harmonized food ontology to increase global food traceability, quality control and data integration](https://www.nature.com/articles/s41538-018-0032-6)
- [Global agricultural concept space: lightweight semantics for pragmatic interoperability](https://www.nature.com/articles/s41538-019-0048-6)

## Meetups, Webinars, Videos

- [Reconciling Disparate Data with Ontologies](https://app.aiplus.training/courses/take/a-data-scientists-rosetta-stone-reconciling-disparate-data-with-ontologies/lessons/22205621-ai-training)

## Knowledge Graph vs Ontology

### What is the difference?

A knowledge base (KB) is fact-oriented but ontology is schema-oriented.

KB: In the Google KGraph, you have certainly a schema (like the one described by DBPedia, Freebase, etc.) and a set of facts (A relation B, "Paris isA City, Paris hasInhabitants 2M, Paris isCapitalOf France, France isA Country, etc.). So, we can (easily) answer to questions like "Number of inhabitants in Paris?" or you can provide a short description of Paris, France (or any given entity in general).

The domain ontology tells people which are the main concepts of a domain, how are these concepts related and which attributes do they have. Here the focus is on the description, with the highest possible expressiveness (disjointness, values restrictions, cardinality restrictions) and the useful annotations (synonyms, definition of terms, examples, comments on design choices, etc.), of the entities of a given domain. Data (or facts) are not the main concern when designing a domain ontology VS KB.

## List of Ontologies

| Ontology | Description |
| :---         |          :--- |
|List of all ontologies in [OLS](https://www.ebi.ac.uk/ols/ontologies)| |
| Google Product Taxonomy | <ul><li>[google_product_category](https://support.google.com/merchants/answer/6324436?hl=en)</li><li>[Visualised](http://vocabulary.semantic-web.at/GoogleProductTaxonomy.html)</li><li>Include [Food, Beverages & Tobacco](http://vocabulary.semantic-web.at/GoogleProductTaxonomy/ba8a279b-5d6e-478b-8963-68348bcab467)</li></ul>|
|[NAPCS – North American Product Classification System](https://www.census.gov/newsroom/press-releases/2020/north-american-product-classification-codes.html)| |
|[Genomic Epidemiology Entity Mart Form Viewer](http://genepio.org/geem/form.html#GENEPIO:0002083)|This draft specification provides a collection of fields related to the contextual data of a specimen, its genomic sequencing, and its pathogenic epidemiology. |

### Food Industry Ontologies

| Ontology | Description |
| :---:        |          :--- |
|[FoodOn – FoodOn: A farm to fork ontology](https://foodon.org/) | FoodOn belongs to the open source [OBO Foundry](http://obofoundry.org/) consortium of interoperable life science oriented ontologies and consequently supports [FAIR](https://en.wikipedia.org/wiki/FAIR_data) data annotation and sharing objectives across a wide variety of academic research, government and commercial sectors. <ul><li>FoodOn reuses terms from OBO Foundry ontologies such as environmental terms from ENVO, agriculture terms from AGRO, plant and animal anatomy terms from UBERON, and PO, organisms from NCBITaxon, relations from RO, and nutritional components from CDNO.</li><li>Conversely, FoodOn terms are reused in a growing list of ontologies such as ENVO, CDNO, ONE, ONS, FIDEO, FOBI, ECTO, and DOID.</li><li>[Integrated Food Ontology Workgroup](https://www.youtube.com/playlist?list=PLhzFEi0G-n-vDmqvPLBinMsoATgyPAzLk)</li><li>[FoodOn](https://github.com/FoodOntology/foodon) GitHub</li><li>[FoodOn](https://www.ebi.ac.uk/ols/ontologies/foodon) on OLS</li><li>[Papers & Articles](https://foodon.org/resources/papers-articles/)</li><li>[Software, Databases and Links](https://foodon.org/resources/software-and-db/)</li><li>[FoodOn Relations](https://foodon.org/design/foodon-relations/)</li></ul>|
|[LanguaL](https://www.langual.org/default.asp)| LanguaL™  -  the International Framework for Food Description. LanguaL™ is a Food Description Thesaurus. LanguaL™ stands for "**Langua** a**L**imentaria" or "language of food". It is an automated method for describing, capturing and retrieving data about food. The work on LanguaL™ was started in the late 1970's by the Center for Food Safety and Applied Nutrition (CFSAN) of the United States Food and Drug Administration as an ongoing co-operative effort of specialists in food technology, information science and nutrition.|
|[The Open Biological and Biomedical Ontology (OBO) Foundry](http://obofoundry.org)| The mission of the OBO Foundry is to develop a family of interoperable ontologies that are both logically well-formed and scientifically accurate. To achieve this, OBO Foundry participants follow and contribute to the development of an evolving set of principles including open use, collaborative development, non-overlapping and strictly-scoped content, and common syntax and relations, based on ontology models that work well, such as the Gene Ontology (GO). |
|[Web Ontology Language (OWL)](https://www.w3.org/OWL/)| The W3C Web Ontology Language (OWL) is a Semantic Web language designed to represent rich and complex knowledge about things, groups of things, and relations between things. OWL is a computational logic-based language such that knowledge expressed in OWL can be exploited by computer programs, e.g., to verify the consistency of that knowledge or to make implicit knowledge explicit. OWL documents, known as ontologies, can be published in the World Wide Web and may refer to or be referred from other OWL ontologies. OWL is part of the W3C’s Semantic Web technology stack, which includes RDF, RDFS, SPARQL, etc.|
|[Food Ontology](https://www.bbc.com/ontologies/fo)|<ul><li>The Food Ontology is a simple lightweight ontology for publishing data about recipes, including the foods they are made from and the foods they create as well as the diets, menus, seasons, courses and occasions they may be suitable for. Whilst it originates in a specific BBC use case, the Food Ontology should be applicable to a wide range of recipe data publishing across the web.</li><li>The Food Ontology sits alongside existing work such as [Google's Rich Snippets for Recipes](https://support.google.com/webmasters/answer/173379?hl=en). While Google, and schema.org, provide a way to represent literal strings in a structured way the Food Ontology provides a richly linked model that more completely describes the recipe and its context. Food Ontology, Google Rich Snippets and Schema.org microdata for recipes are all able to co-exist peacefully within the same site.</li></ul>|


## Knowledge models:

| Model | Description |
| :---:         |          :--- |
|[FOAF](http://xmlns.com/foaf/spec/)| This specification describes the FOAF language, defined as a dictionary of named properties and classes using W3C's RDF technology|
|[Schema.org](https://schema.org)|Schema.org is a collaborative, community activity with a mission to create, maintain, and promote schemas for structured data on the Internet, on web pages, in email messages, and beyond.|
|[SKOS](https://www.ala.org/alcts/resources/z687/skos)| Simple Knowledge Organization System (SKOS) and associated web technologies aim to enable preexisting controlled vocabularies to be consumed on the web and to allow vocabulary creators to publish born-digital vocabularies on the web.

This guide allows catalogers, librarians, and other information professionals to understand and use SKOS, a World Wide Web Consortium (W3C) standard designed for the representation of controlled vocabularies, to be consumed within the web environment.|


## Ready Solutions

| Title | Description |
| :---:         |          :--- |
|[PoolParty Semantic Suite - Your Complete Semantic Platform](https://www.poolparty.biz/) | An example of a comprehensive Semantic Middleware that can integrate with various graph databases. Also, they have a lot of [articles, webinars, etc.](https://www.poolparty.biz/resource-library) on their website. <ul><li>[Document Classification with Semantic AI - Semantic Classifier](https://www.poolparty.biz/semantic-classifier/)</li><li>[PoolParty Semantic Suite - Overview of Applications](https://www.poolparty.biz/product-overview?__hstc=142978346.57c857242b30cb8b2ed4f45839c1f613.1622568285784.1622568285784.1622568285784.1&__hssc=142978346.1.1622568285785&__hsfp=3608221597&hsCtaTracking=ce54104b-d120-4e18-bf85-055fd9fc0b63%7C83e0e358-5ba1-4a2e-a659-5215062b3e3e)</li><li>[Integrations - PoolParty Semantic Suite](https://www.poolparty.biz/integrations)</li><li>How taxonomy can look like (Google Product Taxonomy → [PoolParty Linked Data Server - Taxonomies, Thesauri, Vocabularies](http://vocabulary.semantic-web.at/GoogleProductTaxonomy.visual))</li></ul>|

> Google search: "dish ontology", "ontology vs knowledge graph", "ontology data science", "ontology machine learning", "ontology-based machine learning" 

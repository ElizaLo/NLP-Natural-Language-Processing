# Knowledge Graph

- [Knowledge Graphs](https://paperswithcode.com/task/knowledge-graphs) on PapersWithCode
- [Knowledge Graph](https://stackoverflow.com/search?tab=relevance&q=knowledge%20graph) on StackOverflow

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

| Title | Description |
| :---         |          :--- |
|[Amazon Neptune](https://aws.amazon.com/neptune/?nc1=h_ls )| [Knowledge Graphs on AWS](https://aws.amazon.com/neptune/knowledge-graphs-on-aws/)|
|[GraphDB Downloads and Resources](https://graphdb.ontotext.com/)| Semantic graph databases (also called RDF triplestores)|
|[Neo4j Graph Platform](https://neo4j.com/?_gl=1%2A16rmv03%2A_ga%2AOTM3OTY0NzAxLjE2MjY2OTk4Nzc.%2A_ga_DL38Q8KGQC%2AMTYyNjY5OTg3Ni4xLjEuMTYyNjcwMDA4My4w&_ga=2.37309350.780889525.1626699877-937964701.1626699877)| Neo4j gives developers and data scientists the most trusted and advanced tools to quickly build today’s intelligent applications and machine learning workflows. Available as a fully managed cloud service or self-hosted.|
| [GraphDB™ Ontotext](https://www.ontotext.com/products/graphdb/)| GraphDB allows you to link diverse data, index it for semantic search and enrich it via text analysis to build big knowledge graphs.|
|[AllegroGraph](https://allegrograph.com/products/allegrograph/)| AllegroGraph is a Horizontally Distributed, Multi-model (Document and Graph), Entity-Event Knowledge Graph technology that enables businesses to extract sophisticated decision insights and predictive analytics from their highly complex, distributed data that can’t be answered with conventional databases.|
|[Stardog](https://www.stardog.com)|<ul><li>Stardog is the only graph platform to connect data at the compute layer instead of the storage layer.</li><li>No migrations. No rip and replace. No copies of copies of data. Just a seamless data fabric. </li><li>Stardog is the critical data infrastructure that powers your business’ apps, AI, and analytics.</li></ul>|
|[Eclipse RDF4J](https://rdf4j.org)|Eclipse RDF4J™ is a powerful Java framework for processing and handling RDF data. This includes creating, parsing, scalable storage, reasoning and querying with RDF and Linked Data. It offers an easy-to-use API that can be connected to all leading RDF database solutions. It allows you to connect with SPARQL endpoints and create applications that leverage the power of linked data and Semantic Web.|


## Papers

| Title | Description |
| :---:         |          :--- |
|[Enriching BERT with Knowledge Graph Embeddings for Document Classification](https://arxiv.org/abs/1909.08402) | In this paper, we focus on the classification of books using short descriptive texts (cover blurbs) and additional metadata. Building upon BERT, a deep neural language model, we demonstrate how to combine text representations with metadata and knowledge graph embeddings, which encode author information. Compared to the standard BERT approach we achieve considerably better results for the classification task. For a more coarse-grained classification using eight labels we achieve an F1- score of 87.20, while a detailed classification using 343 labels yields an F1-score of 64.70. We make the source code and trained models of our experiments publicly available.|

## Articles

| Title | Description |
| :---:         |          :--- |
|[Knowledge Graph & NLP Tutorial-(BERT,spaCy,NLTK)](https://www.kaggle.com/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk) |Tutorial |
|[Which is the best tool to build Knowledge Graph or Knowledge Tree?](https://www.quora.com/Which-is-the-best-tool-to-build-Knowledge-Graph-or-Knowledge-Tree)| |
|[Knowledge Graphs with Machine Learning [Guide]](https://neptune.ai/blog/web-scraping-and-knowledge-graphs-machine-learning)| In this article, I’m going to explain how to scrape publicly available data and build knowledge graphs from scraped data, along with some key concepts from Natural Language Processing (NLP). |

## Courses

| Title | Description |
| :---:         |          :--- |
|[CS 520 Knowledge Graphs](https://web.stanford.edu/class/cs520/)| Knowledge graphs have emerged as a compelling abstraction for organizing world's structured knowledge over the internet, capturing relationships among key entities of interest to enterprises, and a way to integrate information extracted from multiple data sources. Knowledge graphs have also started to play a central role in machine learning and natural language processing as a method to incorporate world knowledge, as a target knowledge representation for extracted knowledge, and for explaining what is being learned. This class is a graduate level research seminar and will include lectures on knowledge graph topics (e.g., data models, creation, inference, access) and invited lectures from prominent researchers and industry practitioners. The seminar emphasizes synthesis of AI, database systems and HCI in creating integrated intelligent systems centered around knowledge graphs. |
|[Knowledge Graphs](https://open.hpi.de/courses/knowledgegraphs2020)| In this course you will learn what is necessary to design, implement, and use knowledge graphs. The focus of this course will be on basic semantic technologies including the principles of knowledge representation and symbolic AI. This includes information encoding via RDF triples, knowledge representation via ontologies with OWL, efficiently querying knowledge graphs via SPARQL, latent representation of knowledge in vector space, as well as knowledge graph applications in innovative information systems, as e.g., semantic and exploratory search.|
|[KG Course 2021](https://migalkin.github.io/kgcourse2021/)| Курс лекций по Графам Знаний (Knowledge Graphs). **Graph Representation Learning (GRL)** - одна из самых быстро растущих тем в академическом и деловом сообществах. В настоящее время на русском языке крайне мало структурированной информации и обучающих курсов по основам и использованию Knowledge Graphs (KGs). Мы создали этот курс для всех желающих познакомиться с KGs, релевантными технологиями и перспективными применениями. Концептуально, курс состоит из двух частей - способов работы с KGs. <ul><li>Символьное представление: онтологии, логика, запросы, СУБД;</li><li>Векторное представление: эмбеддинги, graph mining, графовые нейросети, приложения в NLP и Graph ML.</li></ul>|
|[Building Knowledge Graphs with Python](https://www.pluralsight.com/courses/building-knowledge-graphs-python?aid=7010a000002LUv2AAG&promo=&utm_source=non_branded&utm_medium=digital_paid_search_google&utm_campaign=XYZ_EMEA_Dynamic&utm_content=&cq_cmp=1576650371&gclid=Cj0KCQjwktKFBhCkARIsAJeDT0i54h1l8GMTgIT3VpkLsG_tMHDvZE5IbEIDQwm774sOveLHqnu1BRYaAu9yEALw_wcB) | This course will teach you how to create knowledge graphs out of textual information. It will show you how to extract information such as topics and entities and uncover how they are linked into so-called knowledge graphs.|
|[Graph Analytics for Big Data](https://www.coursera.org/learn/big-data-graph-analytics?ranMID=40328&ranEAID=SAyYsTvLiGQ&ranSiteID=SAyYsTvLiGQ-MdBZ_xdvndXDAKb7c3F0qg&siteID=SAyYsTvLiGQ-MdBZ_xdvndXDAKb7c3F0qg&utm_content=10&utm_medium=partners&utm_source=linkshare&utm_campaign=SAyYsTvLiGQ)| Want to understand your data network structure and how it changes under different conditions? Curious to know how to identify closely interacting clusters within a graph? Have you heard of the fast-growing area of graph analytics and want to learn more? This course gives you a broad overview of the field of graph analytics so you can learn new ways to model, store, retrieve and analyze graph-structured data.|
|[Knowledge Graph solution development using TigerGraph](https://www.udemy.com/course/rapid-prototyping-to-build-knowledge-graph-solutions/?utm_source=adwords&utm_medium=udemyads&utm_campaign=LongTail_la.EN_cc.ROW&utm_content=deal4584&utm_term=_._ag_77879423894_._ad_437497333815_._kw__._de_c_._dm__._pl__._ti_dsa-1007766171032_._li_9061018_._pd__._&matchtype=b&gclid=Cj0KCQjwktKFBhCkARIsAJeDT0giKQv3Vpn8d1QOph_XL-awLsL-Re_gl1GQyTzoyea10w1ZkDdRKqIaAsqEEALw_wcB) | "Rapid Prototyping of Knowledge Graph Solutions using TigerGraph" course will help you strategize knowledge graph use cases and help you build or prototype a use case for your knowledge graph engagement.|
|List of Knowledge Graphs  courses |<ul><li>[What online courses explain knowledge graphs in depth?](https://www.quora.com/What-online-courses-explain-knowledge-graphs-in-depth)</li><li>[Ontotext](https://www.ontotext.com/knowledge-hub/videos/)</li></ul> | 

## Libraries

| Title | Description |
| :---:         |          :--- |
|[DEEP GRAPH LIBRARY](https://www.dgl.ai)|<ul><li>Build your models with PyTorch, TensorFlow or Apache MXNet.</li><li>DGL empowers a variety of domain-specific projects including DGL-KE for learning large-scale knowledge graph embeddings, DGL-LifeSci for bioinformatics and cheminformatics, and many others.</li>
</ul> |

## Webinars

- [PoolParty Semantic Classifier - Bringing Machine Learning, NLP and Knowledge Graphs together](https://www.youtube.com/watch?v=SU2to2SVk8M&t=1s)
- [How to Choose The Right Database on AWS](https://www.slideshare.net/ranman96734/how-to-choose-the-right-database-on-aws-berlin-summit-2019/48) - Berlin Summit - 2019


## Ready Solutions

| Title | Description |
| :---:         |          :--- |
|[Google Knowledge Graph Search API](https://developers.google.com/knowledge-graph)| The Knowledge Graph Search API lets you find entities in the Google Knowledge Graph. The API uses standard schema.org types and is compliant with the JSON-LD specification.|

# Knowledge Graph for Recommender Systems

| Title | Description |
| :---:         |          :--- |
|[Knowledge Graph – A Powerful Data Science Technique to Mine Information from Text (with Python code)](https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/)|<ul><li>Knowledge graphs are one of the most fascinating concepts in data science<li><li>Learn how to build a knowledge graph to mine information from Wikipedia pages</li><li>You will be working hands-on in Python to build a knowledge graph using the popular spaCy library</li></ul>|
|[Exploiting Knowledge Graph to Improve Text-based Prediction](https://ieeexplore.ieee.org/document/8622123)|As a special kind of "big data," text data can be regarded as data reported by human sensors. Since humans are far more intelligent than physical sensors, text data contains useful information and knowledge about the real world, making it possible to make predictions about real-world phenomena based on text. As all application domains involve humans, text-based prediction has widespread applications, especially for optimization of decision making. While the problem of text-based prediction resembles text classification when formulated as a supervised learning problem, it is more challenging because the variable to be predicted may not be directly derivable from the text and thus there is a semantic gap between the target variable and the surface features that are often used for representing text data in conventional approaches. In this paper, we propose to bridge this gap by using knowledge graph to construct more effective features for text representation. We propose a two-step filtering algorithm to enhance such a knowledge-aware text representation for a family of entity-centric text regression tasks where the response variable can be treated as an attribute of a group of central entities. We evaluate the proposed algorithm by using two revenue prediction tasks based on reviews. The results show that the proposed algorithm can effectively leverage knowledge graphs to construct interpretable features, leading to significant improvement of the prediction accuracy over traditional features.|
|[Enhancing explanations in recommender systems with knowledge graphs](https://www.sciencedirect.com/science/article/pii/S1877050918316259)|Recommender systems are becoming must-have facilities on e-commerce websites to alleviate information overload and to improve user experience. One important component of such systems is the explanations of the recommendations. Existing explanation approaches have been classified by style and the classes are aligned with the ones for recommendation approaches, such as collaborative-based and content-based. Thanks to the semantically interconnected data, knowledge graphs have been boosting the development of content-based explanation approaches. However, most approaches focus on the exploitation of the structured semantic data to which recommended items are linked (e.g. actor, director, genre for movies). In this paper, we address the under-studied problem of leveraging knowledge graphs to explain the recommendations with items’ unstructured textual description data. We point out 3 shortcomings of the state of the art entity-based explanation approach: absence of entity filtering, lack of intelligibility and poor user-friendliness. Accordingly, 3 novel approaches are proposed to alleviate these shortcomings. The first approach leverages a DBpedia category tree for filtering out incorrect and irrelevant entities. The second approach increases the intelligibility of entities with the classes of an integrated ontology (DBpedia, schema.org and YAGO). The third approach explains the recommendations with the best sentences from the textual descriptions selected by means of the entities. We showcase our approaches within a tourist tour recommendation explanation scenario and present a thorough face-to-face user study with a real commercial dataset containing 1310 tours in 106 countries. We showed the advantages of the proposed explanation approaches on five quality aspects: intelligibility, effectiveness, efficiency, relevance and satisfaction.|
|[Knowledge Graph(KG) for Recommendation System](https://techfirst.medium.com/knowledge-graph-kg-for-recommendation-system-8fe2c6cd354)|This is an introduction on how to integrate Knowledge Graph with Recommendation System.|

# Wikipedia Knowledge Graphs
## Summary
Given a list of topics, ideally Wikipedia pages, we construct and display a knowledge graph of the key entities and their connecting relationships.

## Purpose
Wikipedia is the world's encyclopedia, containing (as far as I'm concerned) the sum total of all human knowledge. Advances in natural language processing techniques 
have created the opportunity to mine this repository of knowledge for patterns and structure representing concepts and information, and embed this data 
into a structured graph that can be queried to answer questions.

## Setup
To set up the environment, we recommend using conda:

```
conda create -n wiki_kg python=3.7
```

Next, activate the environemnt:

```
conda activate wiki_kg
```

And install the following libraries:

```
pip install spacy==2.1.0
pip install neuralcoref
pip install wikipedia-api
pip install networkx
pip install pandas
pip install matplotlib
```

## Results
The output of running the main program should be a graph that looks something like the following:

[KNOWLEDGE GRAPH IMAGE HERE]

## Further Work
- Productionize this by creating a web page where users can select topics from a drop down menu, and render the knowledge graph in their browser
- For large datasets, the results could possibly be improved by using Word2Vec embeddings and cosine similarities to drop irrelevant terms. We could train 
  the model on all the pages used and then drop terms with low co-occurences.
- Something else that could be an interesting side project, perhaps on its own: inferring semantic rules from similar dependency trees. The article (linked 
  below) gives an introduction, but it would be interesting to put some thought into mathematically determining how similar given trees are. This could 
  actually even involve some Case-Based Reasoning:
  - Take some common trees (hyper/hyponyms, etc.) and make rules for them
  - Next, for a sentence that doesn't fit into a pre-defined rule, figure out the most similar dependency tree, and
    modify the rule as needed to fit the new sentence!
  - That rule then goes into our list of rules, and the iteration proceeds
  
  On that note, here's an algorithm for comparing trees: https://arxiv.org/pdf/1508.03381.pdf
  
- Lots of random pages that aren't that relevant get scraped. Can those nodes be dropped later, since in theory they should be "less connected" than the 
  other nodes that are actually "on topic"?

## Sources and Further Reading
- https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/
- https://www.analyticsvidhya.com/blog/2020/06/nlp-project-information-extraction/
- https://www.analyticsvidhya.com/blog/2019/09/introduction-information-extraction-python-spacy/
- https://realpython.com/natural-language-processing-spacy-python/
- https://programmerbackpack.com/python-knowledge-graph-understanding-semantic-relationships/

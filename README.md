# Wikipedia Knowledge Graphs
## Summary

## Purpose
Wikipedia is the world's encyclopedia, containing (approximately) the sum total of all human knowledge. Advances in natural language processing techniques 
have created the opportunity to mine this repository of knowledge for patterns and structure representing concepts and information, and embed this data 
into a structured graph that can be queried to answer questions.

## Usage


## Results and Further Work
- For large datasets, the results could possibly be improved by using Word2Vec embeddings and cosine similarities to drop irrelevant terms. We could train 
  the model on all the pages used and then drop terms with low co-occurences.
- Something else that could be an interesting side project, perhaps on its own: inferring semantic rules from similar dependency trees. The article (linked 
  below) gives an introduction, but it would be interesting to put some thought into mathematically determining how similar given trees are. This could 
  actually even involve some CBR:
  - Take some common trees (hyper/hyponyms, etc.) and make rules for them
  - Next, for a sentence that doesn't fit into a pre-defined rule, figure out the most similar dependency tree, and
    modify the rule as needed to fit the new sentence!
  - That rule then goes into our list of rules, and the iteration proceeds
  
  On that note, here's an algorithm for comparing trees: https://arxiv.org/pdf/1508.03381.pdf

## Sources and Further Reading
- https://www.analyticsvidhya.com/blog/2019/10/how-to-build-knowledge-graph-text-using-spacy/
- https://www.analyticsvidhya.com/blog/2020/06/nlp-project-information-extraction/
- https://www.analyticsvidhya.com/blog/2019/09/introduction-information-extraction-python-spacy/
- https://realpython.com/natural-language-processing-spacy-python/
- https://programmerbackpack.com/python-knowledge-graph-understanding-semantic-relationships/
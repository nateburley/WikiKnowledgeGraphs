########################################################################################################################
## DO THE THING: SCRAPE WIKIPEDIA, APPLY NLP, AND BUILD A KNOWLEDGE GRAPH
# TODO: Lots of random pages that aren't that relevant get scraped. Can those nodes be dropped later, since in theory
#       they should be "less connected" or "weaker connected" than the other nodes that are actually "on topic"?
########################################################################################################################

from wiki_knowledge_graph import *


if __name__ == '__main__':
    # Scrape Wikipedia for a topic
    # wiki_data = wiki_scrape(['Catholic Church', 'Islam', 'Russian Orthodox Church', 'Judaism', 'Buddhism', 'Panpsychism', 'UFO religion'])
    # print("WIKIPEDIA SCRAPE DF LENGTH: {}".format(len(wiki_data.index)))
    # print(wiki_data.head(25))
    # print("\n")

    # # Pickle the wiki_data to not have to scrape a million times
    # datafile = open('religion_wiki_data', 'wb')
    # pickle.dump(wiki_data, datafile)

    # Load in the wiki data, so we don't have to scrape
    infile = open('religion_wiki_data', 'rb')
    wiki_data = pickle.load(infile)

    # Get subject object relationships (which form vertices and edges in the graph)
    # TODO: Parallelize (thread) the shit out of this
    #   - https://realpython.com/intro-to-python-threading/
    all_pairs = extract_all_relations(wiki_data)
    print("ENTITY PAIRS-- SUBJECT/OBJECT RELATIONSHIPS LENGTH: {}".format(len(all_pairs.index)))
    print(all_pairs.head(20))
    print(all_pairs.tail(20))
    print("\n")

    # Draw the graph
    draw_KG(all_pairs)
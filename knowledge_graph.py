from rdflib import Graph, URIRef, Literal


def main():
    # Create a new RDF graph
    g = Graph()

    # Define some entities and relationships
    subject = URIRef("http://example.org/subject")
    predicate = URIRef("http://example.org/predicate")
    object = Literal("Object value")

    # Add a triple to the graph
    g.add((subject, predicate, object))

    # Query the graph
    for s, p, o in g:
        print(f"Subject: {s}, Predicate: {p}, Object: {o}")


if __name__ == "__main__":
    main()

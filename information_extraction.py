# Import the required modules and classes
from sparknlp.annotator import (
    TextMatcher
)
from pyspark.sql.types import StringType

# Import the required modules and classes
from sparknlp.base import DocumentAssembler, Pipeline, ReadAs
from sparknlp.annotator import (
    Tokenizer,
    BigTextMatcher
)


import sparknlp


def main():
    # Start Spark Session
    spark = sparknlp.start()

    texts = ["""As she traveled across the world, Emma visited many different places and
            met many fascinating people. She "
                                  "walked the busy streets of Tokyo, hiked the rugged "
                                  "mountains of Nepal, and swam in the crystal-clear waters
    of the Caribbean. Along the way, she befriended locals like Akira, Rajesh,
    and Maria, each with their own unique stories to tell. Emma's travels took her
    to many cities, including New York, Paris, and Hong Kong, where she savored
    delicious foods and explored vibrant cultures. No matter where she went,
    Emma always found new wonders to discover and memories to cherish."""]

    # Create a dataframe from the sample_text
    data = spark.createDataFrame(texts).toDF("text")

    # PERSON
    person_matches = """
    Emma
    Akira
    Rajesh
    Maria
    """

    with open('data/person_matches.txt', 'w') as f:
        f.write(person_matches)

    # LOCATION
    location_matches = """
    Tokyo
    Nepal
    Caribbean
    New York
    Paris
    Hong Kong
    """

    with open('data/location_matches.txt', 'w') as f:
        f.write(location_matches)


    # Step 1: Transforms raw texts to `document` annotation
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    # Step 2: Gets the tokens of the text
    tokenizer = Tokenizer() \
        .setInputCols("document") \
        .setOutputCol("token")

    # Step 3: PERSON matcher
    person_extractor = TextMatcher() \
        .setInputCols("document", "token") \
        .setEntities("person_matches.txt", ReadAs.TEXT) \
        .setEntityValue("PERSON") \
        .setOutputCol("person_entity") \
        .setCaseSensitive(False)

    # Step 4: LOCATION matcher
    location_extractor = TextMatcher() \
        .setInputCols("document", "token") \
        .setEntities("location_matches.txt", ReadAs.TEXT) \
        .setEntityValue("LOCATION") \
        .setOutputCol("location_entity") \
        .setCaseSensitive(False)


    pipeline = Pipeline().setStages([document_assembler,
                                     tokenizer,
                                     person_extractor,
                                     location_extractor
                                     ])


    # Fit and transform to get a prediction
    results = pipeline.fit(data).transform(data)

    # Display the results
    print(results.selectExpr("person_entity.result").show(truncate=False))

    print(results.selectExpr("location_entity.result").show(truncate=False))

    # Create a dataframe from the sample_text
    data = spark.createDataFrame([
    ["""Natural language processing (NLP) has been a hot topic in recent years,
    with many prominent researchers and practitioners making significant
    contributions to the field. For example, the work of Karen Smith and
    John Johnson on sentiment analysis has helped businesses better understand
    customer feedback and improve their products and services. Meanwhile,
    the research of Dr. Jane Kim on neural machine translation has revolutionized
    the way we approach language translation. Other notable figures in the field
    include Michael Brown, who developed the popular Stanford NLP library, and
    Prof. Emily Zhang, whose work on text summarization has been widely cited
    in academic circles. With so many talented individuals pushing the boundaries
    of what's possible with NLP, it's an exciting time to be involved in this
    rapidly evolving field."""]]).toDF("text")

    # PERSON
    person_matches = """
    Karen Smith
    John Johnson
    Jane Kim
    Michael Brown
    Emily Zhang
    """

    with open('person_matches.txt', 'w') as f:
        f.write(person_matches)


    # Step 1: Transforms raw texts to `document` annotation
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    # Step 2: Gets the tokens of the text
    tokenizer = Tokenizer() \
        .setInputCols("document") \
        .setOutputCol("token")

    # Step 3: PERSON matcher
    person_extractor = BigTextMatcher() \
        .setInputCols("document", "token") \
        .setStoragePath("person_matches.txt", ReadAs.TEXT) \
        .setOutputCol("person_entity") \
        .setCaseSensitive(False)

    pipeline = Pipeline().setStages([document_assembler,
                                     tokenizer,
                                     person_extractor])

    # Fit and transform to get a prediction
    results = pipeline.fit(data).transform(data)

    # Display the results
    print(results.selectExpr("person_entity.result").show(truncate=False))


if __name__ == "__main__":
    main()

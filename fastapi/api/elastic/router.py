from fastapi import APIRouter, Query
import os
import elasticsearch7
from fastapi import APIRouter

# Initialize Elasticsearch client
router = APIRouter()
index = "raw_movies"
es = elasticsearch7(
    hosts=os.getenv("ELASTIC_HOST"),
    http_auth=(os.getenv("ELASTIC_USER"), os.getenv("ELASTIC_PW"))
)

# General Endpoints to support search bar in frontend

# Endpoint for full text search
@router.get("/search/")
async def search(query: str = Query(..., min_length=1, max_length=100)):
    result = es.search(index=index, body={
                       "query": {
                            "match": {
                                "content": query
                                }
                            }
                        })
    hits = result["hits"]["hits"]
    return [hit["_source"] for hit in hits]

# Endpoint for getting suggestions for a partial search query
@router.get("/suggest/title/")
async def suggest(query: str = Query(..., min_length=1, max_length=100)):
    result = es.search(index=index, 
                       body={"suggest": {
                                "text": query, 
                                "movie-suggestion": {
                                    "term": {  
                                        "field": "movie_title"
                                        }
                                    }
                                }})
    suggestions = result["suggest"]["movie-suggestion"][0]["options"]
    return [suggestion["text"] for suggestion in suggestions]

# Endpoint for getting suggestions from multiple fields
@router.get("/suggest/")
async def suggest_multi_field(query: str = Query(..., min_length=1, max_length=100)):
    suggest_body = {
        "suggest": {
            "movie-suggestion": {
                "multi-term": {
                    "fields": ["movie_title", "movie_author"],
                    "prefix_length": 1,
                    "max_edits": 2,
                    "size": 10,
                    "prefix": query
                }
            }
        }
    }
    result = es.search(index=index, body=suggest_body)
    suggestions = result["suggest"]["movie-suggestion"][0]["options"]
    return [suggestion["text"] for suggestion in suggestions]

#Specific endpoints for reveiving single documents

# Endpoint for getting specific movie
@router.get("/movies/{movie_id}")
async def get_by_field_value(movie_id: str):
    query = {"match": {"imdb_id": movie_id}}
    result = es.search(index=index, body={"query": query})
    hits = result["hits"]["hits"]
    if len(hits) == 0:
        return {"message": "No movie found."}
    else:
        return hits[0]["_source"]
    
router = APIRouter()



# # Endpoint for getting the number of documents in the index
# @router.get("/count/")
# async def count():
#     result = es.count(index=index)
#     return result["count"]



# # Endpoint for getting the most frequently occurring terms in the index
# @router.get("/terms/")
# async def terms():
#     result = es.search(index=index, body={"aggs": {"top_terms": {
#                        "terms": {"field": "content.keyword", "size": 10}}}})
#     buckets = result["aggregations"]["top_terms"]["buckets"]
#     return [bucket["key"] for bucket in buckets]
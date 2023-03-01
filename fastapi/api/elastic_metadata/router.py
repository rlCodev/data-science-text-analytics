from fastapi import APIRouter, Query
import os
from dotenv import load_dotenv, find_dotenv
from elasticsearch7 import Elasticsearch
from fastapi import APIRouter
from  config import settings

router = APIRouter()

# Initialize Elasticsearch client
index = "movie_metadata"

es = Elasticsearch(hosts=settings.elastic_host, http_auth=(settings.elastic_user, settings.elastic_pw))

# General Endpoints to support search bar in frontend

# Endpoint for full text search
@router.get("/search")
async def search(query: str = Query(None, min_length=1)):
    if query is None:
        return {"results": []}
    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["original_title", "genres", "tagline", "overview"]
            }
        },
        "sort": [
            {"_score": {"order": "desc"}}
        ],
        "highlight": {
            "fields": {
                "title": {},
                "body": {}
            }
        }
    }
    response = es.search(index=index, body=search_body)
    hits = response["hits"]["hits"]
    results = []
    for hit in hits:
        result = hit["_source"]
        result["id"] = hit["_id"]
        if "highlight" in hit:
            highlights = hit["highlight"]
            if "title" in highlights:
                result["title"] = highlights["title"][0]
            if "body" in highlights:
                result["body"] = highlights["body"][0]
        results.append(result)
    return {"results": results}

# async def search(query: str = Query(None, min_length=1)):
#     if query is None:
#         return {"results": []}
#     search_body = {
#         "query": {
#             "multi_match": {
#                 "query": query,
#                 "fields": ["original_title", "genres", "tagline", "overview"]
#             }
#         },
#         "highlight": {
#             "fields": {
#                 "title": {},
#                 "body": {}
#             }
#         }
#     }
#     response = es.search(index=index, body=search_body)
#     hits = response["hits"]["hits"]
#     results = []
#     for hit in hits:
#         result = hit["_source"]
#         result["id"] = hit["_id"]
#         if "highlight" in hit:
#             highlights = hit["highlight"]
#             if "title" in highlights:
#                 result["title"] = highlights["title"][0]
#             if "body" in highlights:
#                 result["body"] = highlights["body"][0]
#         results.append(result)
#     return {"results": results}




# Endpoint for getting suggestions for a partial search query
@router.get("/suggest/title")
async def suggest(query: str):
    # Set up the Elasticsearch query
    body = {
  "suggest": {
    "title-suggest": {
      "prefix": query,
      "completion": {
        "field": "original_title",
        "size": 10,
        "skip_duplicates": True
      }
    }
  }
}
    
    # Execute the Elasticsearch query
    response = es.search(index=index, body=body)
    
    return response
    # Extract the suggestions from the Elasticsearch response
    suggestions = response["suggest"]["movie-title-suggestion"][0]["options"]
    
    # Format the suggestions as a list of strings
    suggestion_list = [option["text"] for option in suggestions]
    
    # Return the suggestions to the client
    return {"suggestions": suggestion_list}

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


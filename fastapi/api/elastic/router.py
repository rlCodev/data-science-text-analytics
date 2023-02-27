from fastapi import APIRouter
from elasticsearch7

router = APIRouter()

@router.get("/movies/{movie_id}")
def get_movie(movie_id: int):
    return {"movie_id": movie_id}
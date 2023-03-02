import MovieCardDetail from "../components/movie-card-detail";
import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';


export default function Details(profanityCounts) {

  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);


  const { state } = useLocation();
  const  { tmdb_id, imdb_id, original_title, genres, tagline, overview, poster_path, pg_rating, id, profanity_counts } = state;
  return (
    <div className="min-h-screen pt-20 mb-40">
      <div className="bg-white shadow-lg border-gray-100 max-h-400	 border sm:rounded-2xl p-8 flex space-x-8 m-10 flex-wrap z-100">
        <MovieCardDetail></MovieCardDetail>
      </div>
    </div>
  );
};
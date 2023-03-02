import { useLocation } from 'react-router-dom';
import DoughnutChart from "../components/pie-chart";

export default function MovieCardDetail() {
  const { state } = useLocation();
  const  { tmdb_id, imdb_id, original_title, genres, tagline, overview, poster_path, pg_rating, id, profanity_counts } = state;
  // genres = genres.map((genre) => {
  //     return genre.name;
  // });
  return (
    <div className="max-h-90 p-8 flex space-x-8 text-black">
      <div className="h-90 overflow-visible w-1/2">
        <img
          className="rounded-2xl shadow-lg"
          src={`https://www.themoviedb.org/t/p/w600_and_h900_bestv2/${poster_path}`}
          alt={original_title}
        />
      </div>
      <div className="flex flex-col w-1/2 space-y-4">
        <div className="flex justify-between items-start">
          <h2 className="text-2xl font-bold">{original_title}</h2>
          <div className="text-lg bg-yellow-400 font-bold rounded-xl p-2">{pg_rating}</div>
        </div>
        <div>
          <div className="text-sm text-gray-400">Movie</div>
          
            <div className="text-lg text-gray-800">{genres.map((genre) => (<span key={genre.id}>{genre.name} </span>))}</div>
       
        </div>
        <p className="text-lg text-gray-400 max-h-40 overflow-y-hidden">{overview}</p>
        {profanity_counts && <div className="block">
          <h2 className="text-lg text-gray-800 justify-start">Profane Word Frequencies</h2>
          <DoughnutChart profanityCounts={profanity_counts}/>
        </div>}
      </div>
    </div>
  );
}

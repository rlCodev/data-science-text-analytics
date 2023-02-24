import { useNavigate } from 'react-router-dom';

export default function MovieCardDetail({ movie }) {
  const navigate = useNavigate();
  return (
    <div className="max-h-80 p-8 flex space-x-8">
      <div className="h-90 overflow-visible w-1/2">
        <img
          className="rounded-2xl shadow-lg"
          src={`https://www.themoviedb.org/t/p/w600_and_h900_bestv2/${movie.poster_path}`}
          alt={movie.title}
        />
      </div>
      <div className="flex flex-col w-1/2 space-y-4">
        <div className="flex justify-between items-start">
          <h2 className="text-2xl font-bold">{movie.title}</h2>
          <div className="text-lg bg-yellow-400 font-bold rounded-xl p-2">{movie.vote_average}</div>
        </div>
        <div>
          <div className="text-sm text-gray-400">{movie.media_type}</div>
          <div className="text-lg text-gray-800">{movie.release_date}</div>
        </div>
        <p className="text-lg text-gray-400 max-h-40 overflow-y-hidden">{movie.overview}</p>
      </div>
    </div>
  );
}

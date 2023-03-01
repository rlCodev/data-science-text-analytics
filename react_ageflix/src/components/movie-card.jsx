import { useNavigate } from 'react-router-dom';

export default function MovieCard({ movie }) {
  const navigate = useNavigate();
  const { tmdb_id, imdb_id, original_title, genres, tagline, overview, poster_path, pg_rating, id } = movie;

  return (
    <div
      className="w-fit pb-20 hover:cursor-pointer"
      onClick={() => navigate("/details")}
    >
      <div className="transition ease-in-out delay-200 hover:-translate-y-1 hover:scale-105">
        <div className="py-3 ">
          <div className="bg-white shadow-lg border-gray-100 max-h-80 border sm:rounded-2xl p-8 flex space-x-8">
            <div className="h-48 overflow-visible w-1/2">
              <img className="rounded-2xl shadow-lg" src={`https://www.themoviedb.org/t/p/w600_and_h900_bestv2${poster_path}`} alt="" />
              {/* <img className="rounded-2xl shadow-lg"  src="https://www.themoviedb.org/t/p/w600_and_h900_bestv2/1LRLLWGvs5sZdTzuMqLEahb88Pc.jpg" alt=""/> */}
            </div>
            <div className="flex flex-col w-1/2 space-y-4">
              <div className="flex justify-between items-start">
                <h2 className="text-2xl font-bold">{original_title}</h2>
                <div className="text-lg bg-yellow-400 font-bold rounded-xl p-2 min-w-fit">
                  {pg_rating}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-400">Movie</div>
                {/* <div className="text-lg text-gray-800">{releaseYear}</div> */}
              </div>
              <p className="text-lg text-gray-400 max-h-40 overflow-y-hidden">
                {tagline}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

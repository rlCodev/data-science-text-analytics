import DoughnutChart from "../components/pie-chart";
import MovieCardDetail from "../components/movie-card-detail";

export default function Details() {
    return (
      <div className="min-h-screen text-white pt-20 mb-40">
        <div className="bg-white shadow-lg border-gray-100 max-h-400	 border sm:rounded-2xl p-8 flex space-x-8 m-10 flex-wrap">
          <MovieCardDetail></MovieCardDetail>
          <div className="flex flex-row w-1/2 space-y-4">
            <div className="min-w-full"></div>
            <DoughnutChart></DoughnutChart>
          </div>
        </div>
      </div>
    );
};
import { useNavigate } from 'react-router-dom';

export default function MovieCardDetail() {
  const navigate = useNavigate();
    return (

          <div className="max-h-80 p-8 flex space-x-8">
            <div className="h-48 overflow-visible w-1/2">
              <img
                className="rounded-2xl shadow-lg"
                src="https://www.themoviedb.org/t/p/w600_and_h900_bestv2/1LRLLWGvs5sZdTzuMqLEahb88Pc.jpg"
                alt=""
              />
            </div>
            <div className="flex flex-col w-1/2 space-y-4">
              <div className="flex justify-between items-start">
                <h2 className="text-2xl font-bold">Sweet Tooth: El niño ciervo</h2>
                <div className="text-lg bg-yellow-400 font-bold rounded-xl p-2">7.2</div>
              </div>
              <div>
                <div className="text-sm text-gray-400">Series</div>
                <div className="text-lg text-gray-800">2019</div>
              </div>
              <p className="text-lg text-gray-400 max-h-40 overflow-y-hidden">
                Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
                eiusmod tempor incididunt ut labore et dolore magna aliqua.
              </p>
            </div>
          </div>
    );
}
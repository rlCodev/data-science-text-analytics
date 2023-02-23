export default function MovieCard() {
    return (
      <div  class="w-fit pb-20">
      <div class="transition ease-in-out delay-200 hover:-translate-y-1 hover:scale-105">
        <div class="py-3 ">
          <div class="bg-white shadow-lg border-gray-100 max-h-80	 border sm:rounded-2xl p-8 flex space-x-8">
            <div class="h-48 overflow-visible w-1/2">
              <img
                class="rounded-2xl shadow-lg"
                src="https://www.themoviedb.org/t/p/w600_and_h900_bestv2/1LRLLWGvs5sZdTzuMqLEahb88Pc.jpg"
                alt=""
              />
            </div>
            <div class="flex flex-col w-1/2 space-y-4">
              <div class="flex justify-between items-start">
                <h2 class="text-2xl font-bold">Sweet Tooth: El niño ciervo</h2>
                <div class="text-lg bg-yellow-400 font-bold rounded-xl p-2">7.2</div>
              </div>
              <div>
                <div class="text-sm text-gray-400">Series</div>
                <div class="text-lg text-gray-800">2019</div>
              </div>
              <p class="text-lg text-gray-400 max-h-40 overflow-y-hidden">
                Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
                eiusmod tempor incididunt ut labore et dolore magna aliqua.
              </p>
            </div>
          </div>
        </div>
      </div>
      </div>
    );
}
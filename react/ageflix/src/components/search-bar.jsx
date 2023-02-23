export default function SearchBar() {
    return (
      <div className="text-3xl w-full">
        <form className="flex justify-center">
          <div className="max-w-xl">
            <div className="flex space-x-4 min-w-max">
              <div className="flex rounded-md overflow-hidden w-full min-w-[100%]">
                <input
                  type="text"
                  className="w-full rounded-md rounded-r-none font-bold p-2 pl-6 text-lg"
                />
                <button className="bg-yellow-400 px-6 text-lg font-semibold py-4 rounded-r-md">
                  Go
                </button>
              </div>
              <button className="bg-white px-6 text-lg font-semibold py-4 rounded-md">
                Clear
              </button>
            </div>
          </div>
        </form>
      </div>
    );
}
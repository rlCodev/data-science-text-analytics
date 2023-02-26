import { useState, useEffect } from "react";
import SearchBar from "../../components/search-bar";
import MovieCard from "../../components/movie-card";
import Typewriter from 'typewriter-effect';

export default function Search({ movies }) {
  const [searchResults, setSearchResults] = useState([]);

  useEffect(() => {
    // Fetch movie data from API using the search query
    async function fetchMovieData(searchQuery) {
      // const response = await fetch(`https://api.example.com/movies?q=${searchQuery}`);
      const response = await fetch(`test-movies.json`);
      const data = await response.json();
      setSearchResults(data.results);
    }

    // Execute fetchMovieData function with the initial search query
    fetchMovieData("initial search query");
  }, []);

  function handleSearchSubmit(searchQuery) {
    // Fetch movie data from API using the search query
    async function fetchMovieData(searchQuery) {
      // const response = await fetch(`https://api.example.com/movies?q=${searchQuery}`);
      const response = await fetch(`test-movies.json`);
      const data = await response.json();
      setSearchResults(data.results);
    }

    // Execute fetchMovieData function with the new search query
    fetchMovieData(searchQuery);
  }

  return (
    <div className="min-h-screen font-40 pt-40 text-3xl pb-10">
      <h2 className="text-20 text-center font-4 lh-6 ld-04 font-bold text-white mb-6">
        <Typewriter
          options={{
            autoStart: true,
          }}
          onInit={(typewriter) => {
            typewriter
              .typeString("Search for a movie or TV show!")
              .callFunction(() => {})
              .start();
          }}
        />
      </h2>
      <SearchBar onSubmit={handleSearchSubmit} />
      <div className="flex justify-center">
        <div className="grid grid-flow-row-dense grid-cols-2 justify-center p-10 gap-10 max-w-screen-xl pd-y">
          {searchResults.map((movie) => (
            <MovieCard key={movie.id} movie={movie} />
          ))}
        </div>
      </div>
    </div>
  );
};

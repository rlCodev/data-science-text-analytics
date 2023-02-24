import React, { useState, useEffect } from 'react';
import axios from 'axios';
import MovieDisplay from './MovieDisplay';

const MovieList = () => {
  const [movies, setMovies] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      const response = await axios.get('/api/movies');
      setMovies(response.data);
    };
    fetchData();
  }, []);

  return (
    <div>
      <h2>Movie List</h2>
      <MovieDisplay movies={movies} />
    </div>
  );
};

export default MovieList;

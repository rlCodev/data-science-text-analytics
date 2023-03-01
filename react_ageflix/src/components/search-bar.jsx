import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function SearchBar({onSubmit}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [timer, setTimer] = useState(null);

  useEffect(() => {
    if (searchTerm) {
      const fetchData = async () => {
        // const response = await axios.get(`https://example.com/api/search?q=${searchTerm}`);
        const response = await axios.get(`http://127.0.0.1:80/elastic/suggest/title?query=${searchTerm}`);
        const options = response.data.suggest['title-suggest'][0].options;
        const optionTexts = options.map(option => option.text);
        setSuggestions(optionTexts);
        console.log('use effect suggestions', suggestions)
        // todo: replace through api call
      };
      // Wait for 2 seconds before making API call
      const delayTimer = setTimeout(fetchData, 1000);
      setTimer(delayTimer);
    }
    return () => {
      // Clear the timer on unmount or search term change
      clearTimeout(timer);
    };
  }, [searchTerm]);

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(searchTerm);
  };

  const handleChange = event => {
    const { value } = event.target;
    setSearchTerm(value);
    // Clear the timer and start a new one
    clearTimeout(timer);
    const delayTimer = setTimeout(() => {
      const fetchData = async () => {
        const response = await axios.get(`http://127.0.0.1:80/elastic/suggest/title?query=${searchTerm}`);
        const options = response.data.suggest['title-suggest'][0].options;
        const optionTexts = options.map(option => option.text);
        setSuggestions(optionTexts);
        // todo: replace through api call
        console.log(suggestions)
      };
      fetchData();
    }, 5000);
    setTimer(delayTimer);
  };

  const handleSelectSuggestion = (suggestion) => {
    setSearchTerm(suggestion);
    console.log('handleSelectSuggestion', suggestion)
    setSuggestions([]);
  };

  return (
    <div className="text-3xl w-full relative">
      <form onSubmit={handleSubmit} className="flex justify-center">
        <div className="max-w-xl">
          <div className="flex space-x-4 min-w-max">
            <div className="flex rounded-md w-full min-w-[100%] relative">
              <input
                type="text"
                className="w-full rounded-md rounded-r-none font-bold p-2 pl-6 text-lg"
                onChange={handleChange}
                value={searchTerm}
              />
              <button
                className="bg-yellow-400 px-6 text-lg font-semibold py-4 rounded-r-md"
                type="submit"
                onClick={() => {
                  onSubmit(searchTerm);
                }}
              >
                Go
              </button>
              {suggestions.length > 0 && (
              <ul className="absolute top-full left-0 w-full bg-white rounded-md overflow-hidden z-2 mt-1 shadow-yellow-500/50">
                  {suggestions.map((suggestion, index) => (
                    <li
                      key={index}
                      className="font-bold text-lg px-4 py-2 cursor-pointer hover:bg-gray-100 hover: rounded-md"
                      onClick={() => handleSelectSuggestion(suggestion)}
                    >
                      {suggestion}
                    </li>
                  ))}
                </ul>
            )}
            </div>
            <button
              className="bg-white px-6 text-lg font-semibold py-4 rounded-md"
              onClick={() => {
                setSearchTerm('');
                setSuggestions([]);
              }}
            >
              Clear
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}

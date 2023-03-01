import Typewriter from 'typewriter-effect';
import { useNavigate } from 'react-router-dom';

export default function Home() {
  const navigate = useNavigate();
    return (
      <section className="text-gray-600 body-font flex h-screen">
        <div className="max-w-5xl m-auto">
          <h1 className="text-80 text-center font-4 lh-6 ld-04 font-bold text-white mb-6">
          <Typewriter
            options={{
              autoStart: true
            }}
            onInit={(typewriter) => {
              typewriter
                .typeString("Get movie insights using AGEFLIX AI!")
                .callFunction(() => {
                  console.log("String typed out!");
                }).start();
            }}
          />
          </h1>
          <h2 className="text-2xl font-4 font-semibold lh-6 ld-04 pb-11 text-gray-500 text-center">
            Ageflix is a platform using AI to provide you <br/>better insights into content of movies and TV shows.
          </h2>
          <div className="ml-6 text-center">
            <button
              className="rounded-md inline-flex items-center py-3 font-semibold text-black transition duration-500 ease-in-out transform bg-transparent bg-white px-7 text-md md:mt-0 hover:border-red-700 focus:shadow-outline"
              onClick={() => navigate('search')}
            >
              <div className="flex text-lg">
                <span className="justify-center">Get started!</span>
              </div>
            </button>
          </div>
        </div>
      </section>
    );
  }
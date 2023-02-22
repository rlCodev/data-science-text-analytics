import Typewriter from 'typewriter-effect';

export default function Main() {
    return (
      <section class="text-gray-600 body-font min-h-screen">
        <div class="max-w-5xl pt-52 pb-24 mx-auto">
          <h1 class="text-80 text-center font-4 lh-6 ld-04 font-bold text-white mb-6">
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
          <h2 class="text-2xl font-4 font-semibold lh-6 ld-04 pb-11 text-gray-700 text-center">
            Ageflix is a platform using AI to provide you <br/>better insights into content of movies and TV shows.
          </h2>
          <div className="ml-6 text-center">
            <a
              className="inline-flex items-center py-3 font-semibold text-black transition duration-500 ease-in-out transform bg-transparent bg-white px-7 text-md md:mt-0 hover:text-black hover:bg-white focus:shadow-outline"
              href="/"
            >
              <div className="flex text-lg">
                <span className="justify-center">Get started!</span>
              </div>
            </a>
          </div>
        </div>
      </section>
    );
  }
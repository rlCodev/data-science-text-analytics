import Header from "./components/Header";
import Main from "./pages/main";
import Footer from "./components/Footer";

export default function App() {
  return (
    <div className="text-black">
      <Header>
        <title>AGEFLIX</title>
        <link rel="icon" href="/favicon.png" />
      </Header>
      <Main />
      <Footer />
    </div>
  );
}
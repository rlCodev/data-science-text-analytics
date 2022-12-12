import React, { Component } from 'react';
import { BrowserRouter as Router,Routes, Route, Link } from 'react-router-dom';
import './App.css';
  
class App extends Component {
  render() {
    return (
       <Router>
           <Routes>
                 {/* <Route exact path='/' element={< Home />}></Route> */}
          </Routes>
       </Router> 
   );
  }
}
  
export default App;
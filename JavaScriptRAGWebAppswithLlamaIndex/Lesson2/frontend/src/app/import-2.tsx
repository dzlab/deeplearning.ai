"use client";
// import-me: 1
import React, { useState } from 'react';

const QuerySender: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [answer, setAnswer] = useState<string>('');

  const handleSubmit = async (e) => {
    e.preventDefault()
    try {
      const response = await fetch('http://localhost:8000', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Response from the server:', data);
        setAnswer(data.response);
      } else {
        // Handle HTTP errors
        console.error('Server error:', response.status);
      }
    } catch (error) {
      // Handle network errors
      console.error('Network error:', error);
    }
  }

  // Function to update the state with the input value
  const handleChange = (e) => {
    setQuery(e.target.value);
  };  

  return (
    <div>
      <h1>Ask a question</h1>
      <form onSubmit={handleSubmit}>
        <input id="query" type="text" value={query} onChange={handleChange} />
        <button type="submit">Query</button>
      </form>
      <div id="answer">{answer}</div>
    </div>
  );
};

export default QuerySender;

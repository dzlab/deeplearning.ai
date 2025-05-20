"use client";

import React, { useState, useEffect } from "react";

const rubric = [
  { name: "Technical Skills", weight: 0.3 },
  { name: "Communication", weight: 0.25 },
  { name: "Problem Solving", weight: 0.2 },
  { name: "Cultural Fit", weight: 0.15 },
  { name: "Leadership", weight: 0.1 },
];

const FeedbackForm = () => {
  const initialRatings = rubric.reduce((acc, dim) => {
    acc[dim.name] = { rating: "", evidence: "" };
    return acc;
  }, {});

  const [ratings, setRatings] = useState(initialRatings);
  const [totalScore, setTotalScore] = useState(0);
  const [recommendation, setRecommendation] = useState("");
  const [errors, setErrors] = useState({});

  useEffect(() => {
    calculateTotal();
    // eslint-disable-next-line
  }, [ratings]);

  const handleChange = (dim, field, value) => {
    setRatings({
      ...ratings,
      [dim]: {
        ...ratings[dim],
        [field]: value,
      },
    });
  };

  const calculateTotal = () => {
    let total = 0;
    rubric.forEach((dim) => {
      const rating = parseFloat(ratings[dim.name].rating);
      if (!isNaN(rating)) {
        total += rating * dim.weight;
      }
    });
    setTotalScore(total.toFixed(2));
    determineRecommendation(total);
  };

  const determineRecommendation = (score) => {
    if (score >= 4.5) {
      setRecommendation("Strong Hire");
    } else if (score >= 3.5) {
      setRecommendation("Hire");
    } else if (score >= 2.5) {
      setRecommendation("Consider");
    } else {
      setRecommendation("Do Not Hire");
    }
  };

  const validate = () => {
    const newErrors = {};
    rubric.forEach((dim) => {
      if (!ratings[dim.name].rating) {
        newErrors[dim.name] = "Rating is required.";
      }
      if (!ratings[dim.name].evidence.trim()) {
        newErrors[`${dim.name}-evidence`] = "Evidence is required.";
      }
    });
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validate()) {
      // Handle form submission (e.g., send data to server)
      alert("Feedback submitted successfully!");
      setRatings(initialRatings);
      setTotalScore(0);
      setRecommendation("");
      setErrors({});
    }
  };

  const formStyle = {
    maxWidth: "800px",
    margin: "0 auto",
    padding: "20px",
    fontFamily: "Arial, sans-serif",
    color: "#333",
  };

  const sectionStyle = {
    marginBottom: "20px",
    padding: "15px",
    border: "1px solid #ddd",
    borderRadius: "8px",
    backgroundColor: "#f9f9f9",
  };

  const labelStyle = {
    display: "block",
    marginBottom: "8px",
    fontWeight: "bold",
  };

  const inputStyle = {
    width: "100%",
    padding: "8px",
    marginBottom: "10px",
    borderRadius: "4px",
    border: "1px solid #ccc",
  };

  const errorStyle = {
    color: "red",
    fontSize: "0.9em",
    marginBottom: "10px",
  };

  const buttonStyle = {
    padding: "10px 20px",
    backgroundColor: "#28a745",
    color: "#fff",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
  };

  const recommendationStyle = {
    padding: "15px",
    borderRadius: "8px",
    backgroundColor: "#e9ecef",
    fontWeight: "bold",
    textAlign: "center",
  };

  return (
    <form style={formStyle} onSubmit={handleSubmit}>
      <h2>Interview Feedback Form</h2>
      {rubric.map((dim) => (
        <div key={dim.name} style={sectionStyle}>
          <h3>
            {dim.name} (Weight: {dim.weight * 100}%)
          </h3>

          <label style={labelStyle} htmlFor={`${dim.name}-rating`}>
            Rating (1-5):
          </label>
          <select
            id={`${dim.name}-rating`}
            style={inputStyle}
            value={ratings[dim.name].rating}
            onChange={(e) => handleChange(dim.name, "rating", e.target.value)}
          >
            <option value="">Select Rating</option>
            {[1, 2, 3, 4, 5].map((num) => (
              <option key={num} value={num}>
                {num}
              </option>
            ))}
          </select>
          {errors[dim.name] && <div style={errorStyle}>{errors[dim.name]}</div>}

          <label style={labelStyle} htmlFor={`${dim.name}-evidence`}>
            Evidence/Examples:
          </label>
          <textarea
            id={`${dim.name}-evidence`}
            style={inputStyle}
            placeholder={`Provide specific examples demonstrating ${dim.name.toLowerCase()}.`}
            value={ratings[dim.name].evidence}
            onChange={(e) => handleChange(dim.name, "evidence", e.target.value)}
          />
          {errors[`${dim.name}-evidence`] && (
            <div style={errorStyle}>{errors[`${dim.name}-evidence`]}</div>
          )}
        </div>
      ))}

      <div style={sectionStyle}>
        <h3>Total Score: {totalScore}</h3>
        <div style={recommendationStyle}>
          Recommendation: {recommendation || "N/A"}
        </div>
      </div>

      <button type="submit" style={buttonStyle}>
        Submit Feedback
      </button>
    </form>
  );
};

export default FeedbackForm;

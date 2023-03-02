import React from 'react';
import { Bar } from 'react-chartjs-2';
import { ArcElement } from "chart.js";
import Chart from "chart.js/auto";
import { getWordCounts } from '../utils/word-counts';

export default function BarChart({movie}) { 
  const { prediction_frightening, prediction_profanity, prediction_alcohol, prediction_violence, prediction_nudity } = movie;
  return (
  <> 
    <Bar className='max-h-80' data={{
    labels: ["frightening", "profanity", "alcohol", "violence", "nudity"],
    datasets: [
      {
        label: 'Severity of aspects',
        data: [prediction_frightening, prediction_profanity, prediction_alcohol, prediction_violence, prediction_nudity],
        backgroundColor: [
          'rgba(255, 99, 132, 0.2)',
          'rgba(54, 162, 235, 0.2)',
          'rgba(255, 206, 86, 0.2)',
          'rgba(75, 192, 192, 0.2)',
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
        ],
        borderWidth: 1,
      },
    ],
  }} />
  </>
)};
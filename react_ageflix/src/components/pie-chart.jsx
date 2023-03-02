import React from 'react';
import { Doughnut } from 'react-chartjs-2';
import { ArcElement } from "chart.js";
import Chart from "chart.js/auto";
import { getWordCounts } from '../utils/word-counts';

export default function DoughnutChart({profanityCounts}) { 
  let { topWords, topCounts, otherCount } = getWordCounts(profanityCounts, 5);
  topWords.push('Other');
  topCounts.push(otherCount);
  return (
  <> 
    <Doughnut className='max-h-80' data={{
    labels: topWords,
    datasets: [
      {
        label: '# of Votes',
        data: topCounts,
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
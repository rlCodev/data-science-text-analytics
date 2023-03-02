export function getWordCounts(profanityCounts = {}, top=5) {
    // get the counts for all words, sorted in descending order
    const sortedCounts = Object.entries(profanityCounts).sort((a, b) => b[1] - a[1]);
  
    // get the counts and words for the top 5 words
    const topWords = sortedCounts.slice(0, top).map(([word]) => word);
    const topCounts = sortedCounts.slice(0, top).map(([_, count]) => count);
  
    // get the count of all other words
    const otherCount = sortedCounts.slice(5).reduce((acc, [, count]) => acc + count, 0);
  
    return { topWords, topCounts, otherCount };
  }
  
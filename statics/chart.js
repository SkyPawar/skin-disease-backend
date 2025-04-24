fetch("/prediction_stats")
  .then(response => response.json())
  .then(data => {
    const labels = data.map(item => item.time);
    const confidences = data.map(item => {
      const num = parseFloat(item.confidence.replace('%', ''));
      return isNaN(num) ? 0 : num;
    });

    const ctx = document.getElementById("myChart").getContext("2d");
    new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [{
          label: "Prediction Confidence (%)",
          data: confidences,
          fill: false,
          borderColor: "rgb(75, 192, 192)",
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            max: 100
          }
        }
      }
    });
  })
  .catch(error => console.error("Error loading chart:", error));

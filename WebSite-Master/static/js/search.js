const searchButton = document.querySelector("#search-button");

searchButton.addEventListener("click", (e) => {
  const searchText = document.querySelector("#search-bar").value.toLowerCase();
  const rows = document.querySelectorAll("tbody tr");

  rows.forEach((row) => {
    const username = row.querySelector("td:nth-child(1)").textContent.toLowerCase();
    const name = row.querySelector("td:nth-child(2)").textContent.toLowerCase();

    if (username.includes(searchText) || name.includes(searchText)) {
      row.style.display = "";
    } else {
      row.style.display = "none";
    }
  });
});

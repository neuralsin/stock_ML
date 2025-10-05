// static/js/scripts.js

document.addEventListener("DOMContentLoaded", () => {
    const symbolForm = document.getElementById("symbol-form");
    const symbolInput = document.getElementById("symbol-input");
    const symbolList = document.getElementById("symbol-list");
    const runButton = document.getElementById("run-engine");
    const reportTable = document.getElementById("report-table");

    let symbols = [];

    symbolForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const symbol = symbolInput.value.trim().toUpperCase();
        if (symbol && !symbols.includes(symbol)) {
            symbols.push(symbol);
            const li = document.createElement("li");
            li.textContent = symbol;
            symbolList.appendChild(li);
            symbolInput.value = "";
        }
    });

    runButton.addEventListener("click", async () => {
        if (symbols.length === 0) {
            alert("Add at least one symbol to run the engine!");
            return;
        }

        runButton.disabled = true;
        runButton.textContent = "Running...";

        try {
            const response = await fetch("/run_engine", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ symbols })
            });

            const data = await response.json();
            renderReport(data);
        } catch (err) {
            console.error("Error running engine:", err);
            alert("Something went wrong. Check console.");
        } finally {
            runButton.disabled = false;
            runButton.textContent = "Run Engine";
        }
    });

    function renderReport(reportData) {
        reportTable.innerHTML = `
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Total Trades</th>
                    <th>Wins</th>
                    <th>Losses</th>
                    <th>Win Rate</th>
                    <th>Avg PnL</th>
                    <th>Max Drawdown</th>
                    <th>Profit Factor</th>
                </tr>
            </thead>
            <tbody>
                ${reportData.map(row => `
                    <tr>
                        <td>${row.symbol}</td>
                        <td>${row.total_trades}</td>
                        <td>${row.wins}</td>
                        <td>${row.losses}</td>
                        <td>${row.win_rate}</td>
                        <td>${row.avg_pnl}</td>
                        <td>${row.max_drawdown}</td>
                        <td>${row.profit_factor}</td>
                    </tr>
                `).join("")}
            </tbody>
        `;
    }
});

document.addEventListener('DOMContentLoaded', () => {
    const addBtn = document.getElementById('add-stock-btn');
    const runBtn = document.getElementById('run-trading-btn');

    const stockInput = document.getElementById('stock-symbol');
    const tradeSymbolInput = document.getElementById('trade-symbol');
    const intervalSelect = document.getElementById('interval');

    const addMsg = document.getElementById('add-stock-msg');
    const tradeMsg = document.getElementById('trading-msg');
    const tradesOutput = document.getElementById('trades-output');
    const summaryOutput = document.getElementById('summary-output');

    // Add stock
    addBtn.addEventListener('click', async () => {
        const symbol = stockInput.value.trim().toUpperCase();
        if (!symbol) {
            addMsg.textContent = "Please enter a stock symbol.";
            return;
        }

        const res = await fetch('/add_stock', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol})
        });

        const data = await res.json();
        addMsg.textContent = data.message;
        stockInput.value = '';
    });

    // Run trading engine
    runBtn.addEventListener('click', async () => {
        const symbol = tradeSymbolInput.value.trim().toUpperCase();
        const interval = intervalSelect.value;

        if (!symbol) {
            tradeMsg.textContent = "Enter a stock symbol to run trading.";
            return;
        }

        tradeMsg.textContent = "Running trading engine...";
        tradesOutput.innerHTML = '';
        summaryOutput.innerHTML = '';

        const res = await fetch('/run_trading', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol, interval})
        });

        const data = await res.json();

        if (data.success) {
            tradeMsg.textContent = "Trading analysis completed!";
            tradesOutput.innerHTML = `<pre>${JSON.stringify(data.trades, null, 2)}</pre>`;
            summaryOutput.innerHTML = `<pre>${JSON.stringify(data.summary, null, 2)}</pre>`;
        } else {
            tradeMsg.textContent = "Error: " + data.message;
        }
    });
});
